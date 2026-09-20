
import math
from dataclasses import dataclass

import numpy as np

from .laws import deheuvels_ppf
from .tests import folded_critical_value

#: Hard stop on the peeling loop, so a rule that never stops firing cannot hang a run.
MAX_PASSES = 200
#: Reference procedures every comparison carries alongside the gap rules.
BASELINES = ("gumbel",)


def validate_names(names, candidates, complete=False):
    known = set(candidates) | set(BASELINES)
    unknown = sorted(set(names) - known)
    if unknown:
        raise KeyError(f"{unknown}: neither a candidate nor a baseline {list(BASELINES)}")
    if complete:
        missing = sorted(known - set(names))
        if missing:
            raise KeyError(f"{missing}: no entry, though they are live procedures")
    return tuple(names)


def sqrt_2_log(n):
    return math.sqrt(2.0 * math.log(n))


@dataclass(frozen=True)
class GapRule:
    #: "signed" reads both tails of the sorted sample, which is the Renyi test; "upper" reads the
    #: top only, which is the exponential test of prop:exp_test.
    transform: str = "signed"
    width: int = None
    flag: str = "batch"
    max_passes: int = 0
    rescale: bool = False

    @property
    def tails(self):
        return 2 if self.transform == "signed" else 1

    @property
    def name(self):
        width = "full" if self.width is None else f"K{self.width}"
        passes = "" if self.max_passes == 0 else f"_p{self.max_passes}"
        rescale = "_rs" if self.rescale else ""
        return f"{self.transform[:4]}_{width}_{self.flag[:3]}{passes}{rescale}"


def effective_limit(m, width, halve):
    limit = m if width is None else min(width, m)
    if halve:
        limit = min(limit, m // 2 + 1)
    return limit


def side_gaps(v, lo, hi, rule):
    limit = effective_limit(hi - lo, rule.width, rule.tails == 2)
    r = np.arange(1, limit)
    if r.size == 0:
        return np.empty(0), np.empty(0)
    top = v[hi - r] - v[hi - r - 1]
    bottom = v[lo + r] - v[lo + r - 1]
    return top, bottom


def gap_terms(v, lo, hi, rule, scale):
    top, bottom = side_gaps(v, lo, hi, rule)
    if top.size == 0:
        return top, bottom
    return top * scale, bottom * scale


def best_gap(v, lo, hi, rule, scale, critical=None):
    top, bottom = gap_terms(v, lo, hi, rule, scale)
    if top.size == 0:
        return -np.inf, 0, 0

    i_top = int(np.argmax(top))
    i_bot = int(np.argmax(bottom))
    one_tail = rule.tails == 1
    if one_tail or top[i_top] > bottom[i_bot]:
        stat, side = float(top[i_top]), 1
    elif bottom[i_bot] > top[i_top]:
        stat, side = float(bottom[i_bot]), -1
    else:
        stat, side = float(top[i_top]), (1 if i_top <= i_bot else -1)
        if side < 0:
            stat = float(bottom[i_bot])

    if rule.flag == "deepest" and critical is not None:
        crossing_top = np.flatnonzero(top > critical)
        crossing_bot = np.flatnonzero(bottom > critical)
        take_top = int(crossing_top[-1]) + 1 if crossing_top.size else 0
        take_bottom = 0 if one_tail else (int(crossing_bot[-1]) + 1 if crossing_bot.size else 0)
        return stat, take_top, take_bottom

    r_star = (i_top if side > 0 else i_bot) + 1
    take = 1 if rule.flag == "single" else r_star
    return (stat, take, 0) if side > 0 else (stat, 0, take)

def sorted_columns(z):
    return np.sort(z, axis=0)

def gap_statistic_sorted(v, rule, scale):
    n = v.shape[0]
    limit = effective_limit(n, rule.width, rule.tails == 2)
    r = np.arange(1, limit)
    if r.size == 0:
        return np.full(v.shape[1], -np.inf)

    terms = (v[n - r] - v[n - r - 1]) * scale
    if rule.tails == 2:
        terms = np.maximum(terms, (v[r] - v[r - 1]) * scale)
    return terms.max(axis=0)

def gap_statistics_batch(z, rules, scale=None):
    scale = sqrt_2_log(z.shape[0]) if scale is None else scale
    # Every rule now reads the signed sample, so one sort serves all of them.
    v = sorted_columns(z)
    return {name: gap_statistic_sorted(v, rule, scale) for name, rule in rules.items()}


def sorted_cache(z):
    valid = np.where(~np.isnan(z))[0]
    values = z[valid]
    order = np.argsort(values)
    return {"valid": valid, "values": values, "sorted": (order, values[order])}


def flag_gap(z, rule, critical, scale=None, cache=None):
    cache = cache if cache is not None else sorted_cache(z)
    valid, values = cache["valid"], cache["values"]
    m = len(values)
    if m < 2:
        return np.array([], dtype=int), 0

    scale = sqrt_2_log(m) if scale is None else scale
    order, v = cache["sorted"]

    lo, hi = 0, m
    passes = 0
    limit = rule.max_passes or MAX_PASSES
    for _ in range(limit):
        if hi - lo < 2:
            break
        pass_scale = sqrt_2_log(hi - lo) if rule.rescale else scale
        stat, take_top, take_bottom = best_gap(v, lo, hi, rule, pass_scale, critical)
        if stat <= critical or take_top + take_bottom == 0:
            break
        hi -= take_top
        lo += take_bottom
        passes += 1

    flagged = np.concatenate([order[hi:], order[:lo]]).astype(int)
    return valid[flagged], passes


def asymptotic_critical(transform, width, level):
    tails = 2 if transform == "signed" else 1
    terms = 200 if width is None else width - 1
    return deheuvels_ppf(1.0 - level, tails=tails, terms=terms)


def rule_critical(rule, level):
    return asymptotic_critical(rule.transform, rule.width, level)


def gumbel_threshold(n, level):
    return folded_critical_value(n, level)


def flag_gumbel(z, threshold, cache=None):
    cache = cache if cache is not None else sorted_cache(z)
    valid = cache["valid"]
    hits = np.where(np.abs(cache["values"]) > threshold)[0]
    return valid[hits], (1 if hits.size else 0)


def build_procedures(candidates, n, level, criticals, names=None, include_baselines=True,
                     gumbel_level=None):
    names = list(candidates) if names is None else list(names)
    scale = sqrt_2_log(n)
    procedures = {}
    used = {}
    for name in names:
        rule = candidates[name]
        critical = criticals[name] if name in criticals else rule_critical(rule, level)
        used[name] = critical
        procedures[name] = (
            lambda z, cache=None, rule=rule, critical=critical:
                flag_gap(z, rule, critical, scale, cache)
        )

    if include_baselines:
        threshold = gumbel_level if gumbel_level is not None else gumbel_threshold(n, level)
        procedures["gumbel"] = lambda z, cache=None, t=threshold: flag_gumbel(z, t, cache)
        used["gumbel"] = threshold
    return procedures, used

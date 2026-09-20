
import argparse
import json
from pathlib import Path

from .report import write_report

# here because reused multiple times as default in experiments
N_GRID = [200, 352, 619, 1089, 1916, 3372, 5932, 10437, 18362, 32306, 56838, 100000]

def parser(doc, out, n_values=None, n_paths=None, level=0.05, seed=0, from_saved=True):

    p = argparse.ArgumentParser(description=doc)
    if n_values is not None:
        p.add_argument("--n-values", type=int, nargs="+", default=n_values)
    if n_paths is not None:
        p.add_argument("--n-paths", type=int, default=n_paths)
    if level is not None:
        p.add_argument("--level", type=float, default=level)
    if seed is not None:
        p.add_argument("--seed", type=int, default=seed)
    p.add_argument("--out", type=Path, default=None if out is None else Path(out))
    if from_saved:
        p.add_argument("--from-saved", action="store_true")
    return p


class Outputs:
    "For outputting experiment data"

    def __init__(self, args, stem):
        args.out.mkdir(exist_ok=True, parents=True)
        self.directory = args.out
        self.figure = args.out / f"{stem}.png"
        self.json = args.out / f"{stem}.json"
        self.report = args.out / f"{stem}.md"
        self.replotting = getattr(args, "from_saved", False)

    def saved_data(self):
        print(f"-- replotting from {self.json} --", flush=True)
        return json.loads(self.json.read_text())

    def finish(self, figure, title, script, params, doc=None, data=None):
        figure.savefig(self.figure)
        if self.replotting:
            print(f"Saved {self.figure}")
            return
        write_report(self.report, title=title, doc=doc, script=script, params=params, data=data)
        print(f"\nSaved {self.figure}, {self.json} and {self.report}")

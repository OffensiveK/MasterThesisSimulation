
import numpy as np

def normalizing_constants(m):
    log_m = np.log(m)
    b_m = np.sqrt(2 * log_m) - (np.log(np.log(m)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * log_m))
    a_m = 1.0 / np.sqrt(2 * log_m)
    return a_m, b_m


def gumbel_statistic(z):
    valid = np.abs(z[~np.isnan(z)])
    a_m, b_m = normalizing_constants(len(valid))
    return (valid.max() - b_m) / a_m


def gumbel_reject(statistic, level=0.05):
    critical = -np.log(-np.log(1 - level))
    return statistic > critical


def renyi_statistic(z):
    valid = np.abs(z[~np.isnan(z)])
    a_m, _ = normalizing_constants(len(valid))
    m1, m2 = np.partition(valid, -2)[-2:]
    return (max(m1, m2) - min(m1, m2)) / a_m


def renyi_reject(statistic, level=0.05):
    critical = -np.log(level)
    return statistic > critical

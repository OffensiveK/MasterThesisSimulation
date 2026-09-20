
SAFE_BUDGET_BYTES = 1_500_000_000  # 1.5 GB

def chunks(n_paths, batch_size):
    remaining = n_paths
    while remaining > 0:
        take = min(batch_size, remaining)
        yield take
        remaining -= take

def batch_size_for(n, n_paths, arrays_per_path, dtype_bytes=8, budget_bytes=SAFE_BUDGET_BYTES):
    return max(1, min(n_paths, budget_bytes // (dtype_bytes * arrays_per_path * n)))

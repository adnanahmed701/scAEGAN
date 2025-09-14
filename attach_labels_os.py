import os
import time
import numpy as np
import pandas as pd

# -------- settings --------
surv_path = "LUNG_survival.txt"
expr_path = "expression_input.csv"
meth_path = "methylation_input.csv"

expr_out = "scAEGAN/expression_input_OS.csv"
meth_out = "scAEGAN/methylation_input_OS.csv"
random_state = 42
CHUNKSIZE = 50000  # adjust if you have more/less RAM
# --------------------------

def to_tcga12(s: pd.Series | pd.Index) -> pd.Series | pd.Index:
    return s.astype(str).str[:12]

def load_survival(path: str) -> pd.Series:
    """
    Load survival TSV and return a Series:
      index = case_id (TCGA-XX-XXXX)
      values = condition (OS: 0/1)
    Accepts either '_PATIENT' or 'sample' columns to derive case_id.
    """
    s = pd.read_csv(path, sep="\t", dtype=str)
    if "_PATIENT" in s.columns:
        s["case_id"] = s["_PATIENT"]
    elif "sample" in s.columns:
        s["case_id"] = s["sample"].astype(str).str[:12]
    else:
        raise ValueError("Survival file missing _PATIENT or sample column.")

    cond = pd.to_numeric(s.get("OS"), errors="coerce")
    s = s.assign(condition=cond).dropna(subset=["condition"])
    s["condition"] = s["condition"].astype(int)
    s = s[s["condition"].isin([0, 1])]

    # De-dup on case_id; keep first occurrence
    s = s[["case_id", "condition"]].drop_duplicates("case_id")
    s["case_id"] = s["case_id"].astype(str).str[:12]
    s = s.set_index("case_id")
    return s["condition"].astype("int8")

def read_first_col_ids(csv_path: str) -> pd.Index:
    """Read ONLY the first column (IDs) and normalize to TCGA-12."""
    t0 = time.perf_counter()
    first_col = pd.read_csv(csv_path, usecols=[0], dtype=str).iloc[:, 0]
    ids = to_tcga12(first_col).dropna()
    ids = pd.Index(ids.unique())
    print(f"[timing] read IDs from {os.path.basename(csv_path)}: {time.perf_counter()-t0:.2f}s  n={len(ids)}")
    return ids

def balance_ids(cond_map: pd.Series, expr_ids: pd.Index, meth_ids: pd.Index, seed: int) -> pd.Index:
    """Exact 50:50 balance using overlap across expr/meth/survival."""
    common = expr_ids.intersection(meth_ids).intersection(cond_map.index)
    if not len(common):
        raise ValueError("No overlapping case_ids across expression, methylation, and survival.")

    cond_common = cond_map.loc[common]
    counts = cond_common.value_counts()
    print("[info] overlap counts by condition:", counts.to_dict())
    if len(counts) < 2 or counts.min() == 0:
        raise ValueError("Cannot create 50:50 split; one class is missing in the overlap.")

    target_n = int(counts.min())
    rng = np.random.default_rng(seed)
    ids0 = cond_common.index[cond_common == 0]
    ids1 = cond_common.index[cond_common == 1]
    pick0 = rng.choice(ids0, size=target_n, replace=False)
    pick1 = rng.choice(ids1, size=target_n, replace=False)

    balanced = pd.Index(np.concatenate([pick0, pick1]))
    # Deterministic global order (by case_id); both outputs will use this order
    balanced = balanced.sort_values()
    print(f"[info] target_n per class = {target_n}, total balanced IDs = {len(balanced)}")
    return balanced

def collect_filtered_rows(src_csv: str,
                          balanced_ids: pd.Index,
                          cond_map: pd.Series,
                          chunksize: int = 50000) -> pd.DataFrame:
    """
    Stream the big CSV in chunks, keep only rows whose (normalized) index is in balanced_ids,
    attach 'condition', and keep only the first occurrence per case_id.
    Returns a DataFrame indexed by case_id for just the balanced IDs.
    """
    id_set = set(balanced_ids)
    seen = set()
    parts = []
    t0 = time.perf_counter()

    for chunk in pd.read_csv(src_csv, index_col=0, chunksize=chunksize):
        # Normalize index to TCGA-12 and align the chunk with it
        chunk.index = to_tcga12(chunk.index)

        # Filter to balanced IDs
        subset = chunk[chunk.index.isin(id_set)]
        if subset.empty:
            continue

        # Drop duplicates within this subset (keep first)
        subset = subset[~subset.index.duplicated(keep="first")]

        # Drop any IDs we've already taken from previous chunks
        keep_mask = ~subset.index.isin(seen)
        subset = subset[keep_mask]
        if subset.empty:
            continue

        # Attach condition (aligned by index)
        subset["condition"] = cond_map.loc[subset.index].astype("int8").values

        # Mark seen and stash
        seen.update(subset.index)
        parts.append(subset)

        # Early exit if we've collected everything
        if len(seen) == len(balanced_ids):
            break

    if len(seen) < len(balanced_ids):
        missing = set(balanced_ids) - seen
        raise ValueError(f"Some balanced IDs not found in {os.path.basename(src_csv)}: {sorted(list(missing))[:10]} ...")

    out = pd.concat(parts, axis=0)
    # Enforce exact row order across modalities
    out = out.loc[balanced_ids]
    print(f"[timing] collected {len(out)} rows from {os.path.basename(src_csv)} in {time.perf_counter()-t0:.2f}s")
    return out

def atomic_to_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=True)  # index is case_id; we'll reset after
    os.replace(tmp, path)

def main():
    # 1) Labels from survival
    cond_map = load_survival(surv_path)  # Series: index=case_id(12), values=0/1

    # 2) Read ONLY first column (IDs) from big CSVs
    expr_ids = read_first_col_ids(expr_path)
    meth_ids = read_first_col_ids(meth_path)

    # 3) Compute balanced IDs (same for both modalities)
    balanced_ids = balance_ids(cond_map, expr_ids, meth_ids, seed=random_state)

    # 4) Stream-collect rows for each modality and align order
    expr_bal = collect_filtered_rows(expr_path, balanced_ids, cond_map, chunksize=CHUNKSIZE)
    meth_bal = collect_filtered_rows(meth_path, balanced_ids, cond_map, chunksize=CHUNKSIZE)

    # 5) Save (add case_id as column to match your previous outputs)
    expr_bal = expr_bal.reset_index().rename(columns={"index": "case_id"})
    meth_bal = meth_bal.reset_index().rename(columns={"index": "case_id"})

    atomic_to_csv(expr_bal, expr_out)
    atomic_to_csv(meth_bal, meth_out)

    # 6) Sanity: exact balance by unique case_id
    expr_counts = expr_bal.groupby("condition")["case_id"].nunique().to_dict()
    meth_counts = meth_bal.groupby("condition")["case_id"].nunique().to_dict()
    print("\nBalanced datasets saved:")
    print(expr_out, expr_bal.shape)
    print(meth_out, meth_bal.shape)
    print("\nExpression condition counts:", expr_counts)
    print("Methylation condition counts:", meth_counts)

if __name__ == "__main__":
    main()

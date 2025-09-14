import os
import time
import numpy as np
import pandas as pd

# ==================== SETTINGS ====================
clinical_path = "TCGA.LUNG.sampleMap_LUNG_clinicalMatrix.tsv"
expr_path     = "expression_input.csv"
meth_path     = "methylation_input.csv"

expr_out = "scAEGAN/balanced_expression_input.csv"
meth_out = "scAEGAN/balanced_meth_input.csv"

RNG_SEED   = 42
CHUNKSIZE  = 50000   # adjust based on RAM; 50k is usually fine
# ==================================================

def find_gender_column(cols):
    low = [c.lower().strip() for c in cols]
    # prefer exact 'gender', then contains 'gender', then exact 'sex', then contains 'sex'
    for target in ("gender", "sex"):
        for i, c in enumerate(low):
            if c == target:
                return cols[i]
        for i, c in enumerate(low):
            if target in c:
                return cols[i]
    return None

def load_cond_map_from_clinical(path: str) -> pd.Series:
    t0 = time.perf_counter()
    clinical_full = pd.read_csv(path, sep="\t", dtype=str)
    print(f"[timing] read clinical: {time.perf_counter()-t0:.2f}s  shape={clinical_full.shape}")

    if clinical_full.shape[1] < 2:
        raise ValueError("Clinical file must have at least an ID column and one data column.")

    id_col = clinical_full.columns[0]
    gender_col = find_gender_column(clinical_full.columns)
    if gender_col is None:
        raise ValueError(
            "No gender/sex column found in clinical file. "
            f"First columns: {clinical_full.columns.tolist()[:20]}"
        )

    clinical = clinical_full[[id_col, gender_col]].copy().set_index(id_col)
    clinical.index.name = "sampleID"

    map_dict = {"FEMALE": 0, "F": 0, "MALE": 1, "M": 1}
    cond = (clinical[gender_col]
            .astype(str).str.strip().str.upper()
            .map(map_dict).dropna().astype("int8"))
    if cond.empty:
        raise ValueError("No valid gender labels after mapping (expected F/M or Female/Male).")
    return cond

def read_first_col_ids(csv_path: str) -> pd.Index:
    """Read ONLY the first column (sample IDs) from a CSV quickly."""
    t0 = time.perf_counter()
    first_col_df = pd.read_csv(csv_path, usecols=[0], dtype=str)
    # first column values (not index) are the IDs
    ids = first_col_df.iloc[:, 0].astype(str)
    ids = pd.Index(ids.dropna().unique())
    print(f"[timing] read IDs from {os.path.basename(csv_path)}: "
          f"{time.perf_counter()-t0:.2f}s  n_ids={len(ids)}")
    return ids

def balance_ids(cond_map: pd.Series, expr_ids: pd.Index, meth_ids: pd.Index, seed: int) -> pd.Index:
    """Compute exact 50:50 balanced IDs from overlaps only."""
    common_ids = expr_ids.intersection(meth_ids).intersection(cond_map.index)
    if len(common_ids) == 0:
        raise ValueError("No overlapping sampleIDs across expression/methylation and clinical labels.")

    cond_common = cond_map.loc[common_ids]
    counts = cond_common.value_counts()
    print("[info] overlap counts by condition:", counts.to_dict())
    if len(counts) < 2 or counts.min() == 0:
        raise ValueError("Cannot create a 50:50 split; one class missing in overlap.")

    target_n = int(counts.min())
    rng = np.random.default_rng(seed)
    ids0 = cond_common.index[cond_common == 0]
    ids1 = cond_common.index[cond_common == 1]
    pick0 = rng.choice(ids0, size=target_n, replace=False)
    pick1 = rng.choice(ids1, size=target_n, replace=False)

    balanced = pd.Index(np.concatenate([pick0, pick1]))
    # Preserve a deterministic global order (by ID); change to your preference if needed
    balanced = balanced.sort_values()
    print(f"[info] target_n per class = {target_n}, total balanced IDs = {len(balanced)}")
    return balanced

def write_filtered_with_order(src_csv: str, out_csv: str,
                              balanced_ids: pd.Index,
                              cond_map: pd.Series,
                              chunksize: int = 50000) -> None:
    """
    Stream the source CSV in chunks, keep only balanced_ids, append 'condition',
    sort each chunk by the global order, and write out incrementally.
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Fast membership tests + global order
    id_set   = set(balanced_ids)
    ordermap = {sid: i for i, sid in enumerate(balanced_ids)}

    # We read with index_col=0 so the first column (sampleID) becomes the index
    first = True
    total_rows = 0
    t0 = time.perf_counter()
    for chunk in pd.read_csv(src_csv, index_col=0, chunksize=chunksize):
        # Keep only rows in our balanced set
        subset = chunk[chunk.index.isin(id_set)]
        if subset.empty:
            continue

        # Append condition from cond_map (aligned by index)
        subset["condition"] = cond_map.loc[subset.index].astype("int8").values

        # Sort rows by the global order so this file matches the other modality
        subset["__rank__"] = subset.index.map(ordermap)
        subset = subset.sort_values("__rank__").drop(columns="__rank__")

        # Write (header once, then append)
        subset.to_csv(out_csv, mode="w" if first else "a", header=first)
        first = False
        total_rows += len(subset)

    print(f"[timing] wrote {total_rows} rows to {out_csv} in {time.perf_counter()-t0:.2f}s")

def main():
    # 1) Build condition map from clinical
    cond_map = load_cond_map_from_clinical(clinical_path)

    # 2) Read ONLY first column (IDs) from big CSVs
    expr_ids = read_first_col_ids(expr_path)
    meth_ids = read_first_col_ids(meth_path)

    # 3) Compute balanced IDs
    balanced_ids = balance_ids(cond_map, expr_ids, meth_ids, seed=RNG_SEED)

    # 4) Stream-filter each CSV and write outputs with identical row order + condition
    write_filtered_with_order(expr_path, expr_out, balanced_ids, cond_map, chunksize=CHUNKSIZE)
    write_filtered_with_order(meth_path, meth_out, balanced_ids, cond_map, chunksize=CHUNKSIZE)

    # 5) Sanity: print final balance
    counts = pd.Series(cond_map.loc[balanced_ids]).value_counts().to_dict()
    print("Balanced condition counts (0=female, 1=male):", counts)
    print("Outputs:", expr_out, meth_out)

if __name__ == "__main__":
    main()

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

def parse_args():
    p = argparse.ArgumentParser(description="Process parquet + reads, filter, and save outputs")
    p.add_argument("--parquet_dir", default="outputs/generate-parquet/parquet",
                   help="Directory with parquet files (default: outputs/generate-parquet/parquet)")
    p.add_argument("--reads_dir",   default="outputs/generate-parquet/sandpiper",
                   help="Directory with Sandpiper TSVs (default: outputs/generate-parquet/sandpiper)")
    p.add_argument("--rank",        default="s__", choices=["s__","g__","f__","o__","c__","p__"],
                   help="Taxonomic rank (default: s__)")
    p.add_argument("--out_dir",     default="outputs/process-data",
                   help="Directory to write processed outputs (default: outputs/process-data)")
    return p.parse_args()

def load_parquets(parquet_dir: str, rank: str):
    genes_pq = os.path.join(parquet_dir, "gene_coverage.parquet")
    taxon_pq = os.path.join(parquet_dir, f"taxon_coverage_{rank.replace('__','')}.parquet")
    if not os.path.isfile(genes_pq):
        raise FileNotFoundError(f"Missing {genes_pq}")
    if not os.path.isfile(taxon_pq):
        raise FileNotFoundError(f"Missing {taxon_pq}")
    df_genes = pd.read_parquet(genes_pq)
    df_species = pd.read_parquet(taxon_pq)
    shared = df_genes.index.intersection(df_species.index)
    if shared.empty:
        raise ValueError("No overlapping samples between gene and taxon parquet indices")
    df_genes = df_genes.loc[shared]
    df_species = df_species.loc[shared]
    return df_genes, df_species

def extract_total_reads(reads_dir: str) -> pd.Series:
    pat = re.compile(r"total mapped reads:\s*([0-9.eE+-]+)")
    totals = {}
    for fn in tqdm(os.listdir(reads_dir), desc="Scanning total reads"):
        fp = os.path.join(reads_dir, fn)
        if not os.path.isfile(fp): 
            continue
        with open(fp, "r") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    try:
                        totals[os.path.splitext(fn)[0]] = float(m.group(1))
                    except ValueError:
                        pass
                    break
    if not totals:
        raise ValueError(f"No 'total mapped reads' entries found in {reads_dir}")
    return pd.Series(totals)

def logistic(x, A, k, x0):
    return A / (1 + np.exp(-k * (np.log10(x) - x0)))

def logistic_deriv_log10(x, A, k, x0):
    e = np.exp(-k * (np.log10(x) - x0))
    return (A * k * e) / (1 + e) ** 2

def saturated_filter(df_genes: pd.DataFrame, total_reads_ser: pd.Series, out_plot: str):
    genes_per_sample = (df_genes > 0).astype(int).sum(axis=1)
    common = df_genes.index.intersection(total_reads_ser.index)
    if common.empty:
        raise ValueError("No overlap between df_genes and total_reads_ser")
    x_reads = total_reads_ser.loc[common].values
    y_genes = genes_per_sample.loc[common].values

    popt, _ = curve_fit(
        logistic, x_reads, y_genes,
        p0=[y_genes.max(), 1, np.log10(np.median(x_reads))]
    )
    x_fit = np.logspace(np.log10(x_reads.min()), np.log10(x_reads.max()), 10_000)
    dy_dx = logistic_deriv_log10(x_fit, *popt)
    max_slope = dy_dx.max()
    slope_threshold = max_slope * 1_000
    below = np.where(dy_dx < slope_threshold)[0]
    reads_threshold = x_fit[below[0]] if below.size else x_fit[-1]
    reads_threshold = max(reads_threshold, 8e5)

    plt.figure(figsize=(5,4))
    hb = plt.hexbin(x_reads, y_genes, gridsize=60, bins="log", xscale="log", yscale="log")
    plt.plot(x_fit, logistic(x_fit, *popt), lw=2, label="Logistic fit")
    plt.axvline(reads_threshold, ls="--", label=f"Threshold ≈ {reads_threshold:.1e}")
    plt.colorbar(hb, label="N samples")
    plt.ylim(1e1)
    plt.xlabel("Total reads")
    plt.ylabel("Genes detected")
    plt.title("Genes detected vs total reads")
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(out_plot, bbox_inches="tight")
    plt.close()

    sat_mask = (total_reads_ser >= reads_threshold).reindex(df_genes.index, fill_value=False)
    return sat_mask, reads_threshold

def threshold_features(df_genes: pd.DataFrame, df_species: pd.DataFrame, min_prop: float = 0.10):
    n = df_genes.shape[0]
    thr = int(min_prop * n)
    count_sp   = (df_species > 0).sum(axis=0)
    count_gene = (df_genes   > 0).sum(axis=0)
    df_species_f = df_species.loc[:, (count_sp   >= thr) & (count_sp   < n)]
    df_genes_f   = df_genes  .loc[:, (count_gene >= thr) & (count_gene < n)]
    return df_genes_f, df_species_f

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=== Loading parquet tables ===")
    df_genes, df_species = load_parquets(args.parquet_dir, args.rank)

    print("=== Extracting total mapped reads ===")
    total_reads = extract_total_reads(args.reads_dir)

    print("=== Estimating saturation threshold and filtering samples ===")
    plot_path = os.path.join(args.out_dir, "saturation_plot.pdf")
    sat_mask, thr = saturated_filter(df_genes, total_reads, plot_path)
    df_genes = df_genes.loc[sat_mask]
    df_species = df_species.loc[sat_mask]

    print("=== Thresholding features (10%) ===")
    df_genes_f, df_species_f = threshold_features(df_genes, df_species, min_prop=0.10)

    print("=== Saving filtered outputs ===")
    df_genes_f.to_parquet(os.path.join(args.out_dir, "genes_filtered.parquet"))
    df_species_f.to_parquet(os.path.join(args.out_dir, "species_filtered.parquet"))

    sat_n = int(sat_mask.sum()); total = int(len(sat_mask))
    print(f"Saturation threshold: {thr:.2e} reads")
    print(f"Saturated samples:   {sat_n}/{total} ({100*sat_n/total:.2f}%)")
    print(f"Plot saved to: {plot_path}")
    print(f"Filtered genes parquet:   {os.path.join(args.out_dir,'genes_filtered.parquet')}")
    print(f"Filtered species parquet: {os.path.join(args.out_dir,'species_filtered.parquet')}")

if __name__ == "__main__":
    main()

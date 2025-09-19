import os
import argparse
from typing import Dict, Iterable

import pandas as pd
from tqdm import tqdm

VALID_TAXON_RANKS = ["s__", "g__", "f__", "o__", "c__", "p__"]


def parse_args():
    p = argparse.ArgumentParser(description="Prepare parquet data")
    p.add_argument("--genes", action="store_true", help="Generate gene coverage parquet")
    p.add_argument("--gene_reads", action="store_true", help="Generate gene read counts parquet")
    p.add_argument("--taxon", type=str, choices=VALID_TAXON_RANKS + ["all"], default=None,
                   help="Generate taxon coverage parquet at rank")
    p.add_argument("--exclude_taxa", type=str, nargs="+", default=None, help="Taxa to exclude")
    p.add_argument("--species_dir", type=str, help="Path to species input directory")
    p.add_argument("--genes_dir", type=str, help="Path to genes input directory")
    p.add_argument("--data_dir", type=str, default="data", help="Directory to store outputs")
    p.add_argument("--verbose", action="store_true", help="Print verbose debugging information")
    return p.parse_args()


# ---------------------------- Gene processing ----------------------------

def _process_gene_file(file_path: str, extract_reads: bool = False) -> Dict[str, float]:
    gene_dict: Dict[str, float] = {}
    col_idx = 2 if extract_reads else 1  # 0: gene, 1: coverage, 2: n_reads

    try:
        with open(file_path, "r") as f:
            found_header = False
            for line in f:
                if line.startswith("#"):
                    if "\tgene\tcoverage\tn_reads" in line:
                        found_header = True
                    continue
                if not found_header:
                    continue
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    gene = parts[0]
                    try:
                        val = float(parts[col_idx])
                    except ValueError:
                        continue
                    gene_dict[gene] = val
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}
    return gene_dict


def build_gene_parquet(genes_dir: str, output_path: str, extract_reads: bool = False):
    if not os.path.isdir(genes_dir):
        raise FileNotFoundError(f"genes_dir does not exist: {genes_dir}")

    files = [f for f in os.listdir(genes_dir) if os.path.isfile(os.path.join(genes_dir, f))]
    if not files:
        print(f"No gene files found in '{genes_dir}'")
        return

    label = "Read Counts" if extract_reads else "Coverage"
    print(f"\n=== Generating Gene {label} Parquet ===")

    all_data: Dict[str, Dict[str, float]] = {}
    for fname in tqdm(files, desc="Processing samples"):
        sample_id = os.path.splitext(fname)[0]
        fpath = os.path.join(genes_dir, fname)
        d = _process_gene_file(fpath, extract_reads=extract_reads)
        if d:
            all_data[sample_id] = d

    if not all_data:
        print("No gene data parsed; nothing to write.")
        return

    df = pd.DataFrame.from_dict(all_data, orient="index").fillna(0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=True)
    print(f"✅ Saved gene {label.lower()} to {output_path} ({df.shape[0]} samples × {df.shape[1]} genes)")


# --------------------------- Taxon processing ----------------------------

def _excluded(tokens: Iterable[str], exclusions: Iterable[str]) -> bool:
    for t in tokens:
        if "__" in t:
            parts = t.split("__", 1)
            if len(parts) == 2 and parts[1] in exclusions:
                return True
    return False


def _process_taxon_file(file_path: str, rank: str, exclusions: Iterable[str], verbose=False) -> Dict[str, float]:
    cov: Dict[str, float] = {}
    try:
        with open(file_path, "r") as fh:
            if verbose:
                print(f"Processing {file_path} for rank {rank}")
                header = next(fh, None)
                print(f"Header: {header}")
                fh.seek(0)  # Reset to beginning

            next(fh, None)  # skip header
            for i, line in enumerate(fh):
                if verbose and i < 3:
                    print(f"Sample line {i+1}: {line.strip()}")

                cols = line.strip().split("\t")
                if len(cols) < 3:
                    if verbose and i < 3:
                        print(f"  - Not enough columns: {len(cols)}")
                    continue

                try:
                    coverage = float(cols[1])
                except ValueError:
                    if verbose and i < 3:
                        print(f"  - Invalid coverage: {cols[1]}")
                    continue

                taxonomy_str = cols[2]
                tokens = [t.strip() for t in taxonomy_str.split(";") if t.strip()]

                if verbose and i < 3:
                    print(f"  - Taxonomy tokens: {tokens}")

                if exclusions and _excluded(tokens, exclusions):
                    continue

                found = False
                for token in tokens:
                    if token.startswith(rank):
                        name = token[len(rank):].strip()
                        cov[name] = cov.get(name, 0.0) + coverage
                        found = True
                        break

                if verbose and i < 3 and not found:
                    print(f"  - No match for rank {rank}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return {}

    if verbose:
        print(f"Found {len(cov)} taxa at rank {rank} in {file_path}")

    return cov


def build_taxon_parquet(species_dir: str, rank: str, output_path: str, exclusions=None, verbose=False):
    if not os.path.isdir(species_dir):
        raise FileNotFoundError(f"species_dir does not exist: {species_dir}")

    files = [f for f in os.listdir(species_dir) if os.path.isfile(os.path.join(species_dir, f))]
    if not files:
        print(f"No species files found in '{species_dir}'")
        return

    print("\n=== Generating Taxonomic Coverage Parquet ===")
    print(f"\nProcessing taxonomic rank: {rank}")

    all_data: Dict[str, Dict[str, float]] = {}
    for fname in tqdm(files, desc="Processing samples"):
        sample_id = os.path.splitext(fname)[0]
        fpath = os.path.join(species_dir, fname)
        d = _process_taxon_file(fpath, rank=rank, exclusions=exclusions or [], verbose=verbose)
        if d:
            all_data[sample_id] = d

    if not all_data:
        print("No taxon data parsed; nothing to write.")
        return

    df = pd.DataFrame.from_dict(all_data, orient="index").fillna(0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=True)
    print(f"✅ Saved taxon coverage ({rank}) to {output_path} ({df.shape[0]} samples × {df.shape[1]} taxa)")


# --------------------------------- CLI ----------------------------------

def main():
    args = parse_args()

    if not (args.genes or args.gene_reads or args.taxon):
        print("⚠️ Nothing to do. Use --genes, --gene_reads and/or --taxon.")
        return

    if (args.genes or args.gene_reads) and not args.genes_dir:
        raise ValueError("--genes_dir is required when using --genes or --gene_reads")
    if args.taxon and not args.species_dir:
        raise ValueError("--species_dir is required when using --taxon")

    data_dir = os.path.abspath(args.data_dir)
    parquet_dir = os.path.join(data_dir, "parquet")
    os.makedirs(parquet_dir, exist_ok=True)

    if args.genes:
        out = os.path.join(parquet_dir, "gene_coverage.parquet")
        build_gene_parquet(args.genes_dir, out, extract_reads=False)

    if args.gene_reads:
        out = os.path.join(parquet_dir, "gene_reads.parquet")
        build_gene_parquet(args.genes_dir, out, extract_reads=True)

    if args.taxon:
        ranks = VALID_TAXON_RANKS if args.taxon == "all" else [args.taxon]
        for r in ranks:
            out = os.path.join(parquet_dir, f"taxon_coverage_{r.replace('__','')}.parquet")
            build_taxon_parquet(args.species_dir, r, out, exclusions=args.exclude_taxa, verbose=args.verbose)

    print("\n✅ Data preparation complete!")
    print(f"Parquet directory: {parquet_dir}")


if __name__ == "__main__":
    main()

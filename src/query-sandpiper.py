#!/usr/bin/env python3
import os
import argparse
import asyncio
import aiohttp
from tqdm import tqdm

API_FMT = "https://sandpiper.qut.edu.au/api/condensed_csv/{run}"

async def _fetch_one(run, session, out_dir, sem, pbar):
    url = API_FMT.format(run=run)
    try:
        async with sem:
            async with session.get(url) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                text = await r.text()
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{run}.tsv"), "w") as f:
            f.write(text)
    except Exception as e:
        print(f"\nFailed {run}: {e}")
    finally:
        pbar.update(1)

async def fetch_condensed_async(input_dir, output_dir, concurrency=20, timeout_sec=60, max_n=0):
    # Collect run IDs from filenames (accept .csv or .tsv, use the stem as run)
    runs = [os.path.splitext(f)[0] for f in os.listdir(input_dir)
            if f.lower().endswith((".csv", ".tsv"))]
    if max_n:
        runs = runs[:max_n]

    os.makedirs(output_dir, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=timeout_sec)
    connector = aiohttp.TCPConnector(ssl=False)  # match your working setup

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        with tqdm(total=len(runs), desc="Fetching from Sandpiper") as pbar:
            tasks = [_fetch_one(run, session, output_dir, sem, pbar) for run in runs]
            await asyncio.gather(*tasks)

def main():
    ap = argparse.ArgumentParser(
        description="Fetch condensed TSVs from Sandpiper for runs listed by filenames in a directory."
    )
    ap.add_argument("input_dir", help="Directory containing .csv/.tsv files (filenames are run accessions).")
    ap.add_argument("output_dir", help="Directory to write <run>.tsv files.")
    ap.add_argument("--conc", type=int, default=20, help="Concurrency (default: 20).")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default: 60).")
    ap.add_argument("--max", type=int, default=0, help="Limit number of runs (default: 0 = no limit).")
    args = ap.parse_args()

    asyncio.run(fetch_condensed_async(
        args.input_dir, args.output_dir, concurrency=args.conc,
        timeout_sec=args.timeout, max_n=args.max
    ))

if __name__ == "__main__":
    main()

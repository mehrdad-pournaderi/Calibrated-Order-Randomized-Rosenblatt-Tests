#!/usr/bin/env bash
# Re-download the nine H.10 exchange-rate series from FRED into ../data/.
# Retrieved for the paper on 2026-07-15.
set -euo pipefail
mkdir -p ../data
for id in DEXUSEU DEXJPUS DEXUSUK DEXCAUS DEXSZUS DEXUSAL DEXUSNZ DEXSDUS DEXNOUS; do
  echo "fetching $id"
  curl -s --max-time 60 \
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=${id}&cosd=2015-01-01&coed=2025-12-31" \
    -o "../data/fx_${id}.csv"
done
echo "done"

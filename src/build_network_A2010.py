"""
build_network_A2010.py  —  P4 Network Pruning Task
====================================================
Steps:
  1. Smoke-test the proximity data (phi range, symmetry, product-set match).
  2. Prune via maximum spanning tree + phi >= 0.4 edges.
  3. Generate a reproducible spring layout (fixed seed).
  4. Attach product metadata (name, sector, pci) to every node.
  5. Save nodes + edges as network_A2010.parquet (two DataFrames in one file
     is awkward, so we save two parquet files:
       outputs/network_A2010_nodes.parquet
       outputs/network_A2010_edges.parquet
     and also a combined long-format sentinel file: network_A2010.parquet
     whose 'table' column tags each section).
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import networkx as nx

# ------------------------------------------------------------------ paths ---
ROOT        = Path(__file__).resolve().parents[1]
PROX_PATH   = ROOT / "p2" / "data-p2" / "proximity" / "proximity_A2010.parquet"
PROD_PATH   = ROOT / "data" / "processed" / "products_v1.parquet"
OUT_DIR     = ROOT / "outputs" / "network"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NODES   = OUT_DIR / "network_A2010_nodes.parquet"
OUT_EDGES   = OUT_DIR / "network_A2010_edges.parquet"

LAYOUT_SEED       = 42
PHI_PRUNE_THRESH  = 0.40

# =========================================================== 1. Load data ===
print("=" * 70)
print("STEP 1 — LOAD & SMOKE TEST")
print("=" * 70)

print(f"\nLoading proximity: {PROX_PATH.name} ...")
prox = pd.read_parquet(PROX_PATH)
print(f"  Rows loaded: {len(prox):,}  |  Columns: {prox.columns.tolist()}")

print(f"\nLoading products:  {PROD_PATH.name} ...")
prod = pd.read_parquet(PROD_PATH)
print(f"  Rows loaded: {len(prod):,}  |  Columns: {prod.columns.tolist()}")

# ============================================== 1a. Phi range [0, 1] check ===
phi_min = prox["phi"].min()
phi_max = prox["phi"].max()
phi_ok  = (phi_min >= 0.0) and (phi_max <= 1.0)
print(f"\n[Smoke test] phi range: min={phi_min:.6f}, max={phi_max:.6f}  → {'PASS' if phi_ok else 'FAIL'}")
if not phi_ok:
    print("  ERROR: phi values outside [0, 1]!", file=sys.stderr)
    sys.exit(1)

# ================================================ 1b. Symmetry check ===
# The parquet stores the full n×n matrix (including self-loops).
# Check that phi(i,j) == phi(j,i) for every pair.
prox_no_self = prox[prox["product_i"] != prox["product_j"]].copy()

# Build a lookup dict for fast symmetric check on a sample
# (full O(n²) merge can be slow; we'll pivot a sample and also do a full merge)
print(f"\n[Smoke test] Checking symmetry via merge on all {len(prox_no_self):,} off-diagonal edges ...")
prox_swap = prox_no_self.rename(
    columns={"product_i": "product_j", "product_j": "product_i", "phi": "phi_swap"}
)
merged = prox_no_self.merge(prox_swap, on=["product_i", "product_j"], how="inner")
sym_ok = np.allclose(merged["phi"].values, merged["phi_swap"].values, atol=1e-9)
print(f"  Symmetric pairs checked: {len(merged):,}  → {'PASS' if sym_ok else 'FAIL'}")
if not sym_ok:
    mismatch = merged[~np.isclose(merged["phi"], merged["phi_swap"], atol=1e-9)]
    print(f"  FAIL: {len(mismatch)} asymmetric pairs found.")
    print(mismatch.head(5))

# ======================================== 1c. Product-set match check ===
prox_products = set(prox["product_i"].unique()) | set(prox["product_j"].unique())
prod_products  = set(prod["product"].astype(str).str.strip().str.upper())
in_prox_not_prod = prox_products - prod_products
in_prod_not_prox = prod_products - prox_products
match_ok = (len(in_prox_not_prod) == 0) and (len(in_prod_not_prox) == 0)
print(f"\n[Smoke test] Product-set match:")
print(f"  Products in products_v1:   {len(prod_products):,}")
print(f"  Products in proximity:     {len(prox_products):,}")
print(f"  In proximity not in products_v1: {len(in_prox_not_prod)}")
print(f"  In products_v1 not in proximity: {len(in_prod_not_prox)}")
print(f"  → {'PASS' if match_ok else 'WARN (mismatch)'}")
if in_prox_not_prod:
    print(f"    Extra in proximity: {sorted(in_prox_not_prod)[:10]} {'...' if len(in_prox_not_prod)>10 else ''}")
if in_prod_not_prox:
    print(f"    Missing from proximity: {sorted(in_prod_not_prox)[:10]} {'...' if len(in_prod_not_prox)>10 else ''}")

# =========================================== 1d. Edge count before pruning ===
# Use only the upper triangle (product_i < product_j) to avoid double-counting.
upper = prox_no_self[prox_no_self["product_i"] < prox_no_self["product_j"]].copy()
n_edges_before = len(upper)
print(f"\nEdge count BEFORE pruning (undirected, excluding self-loops): {n_edges_before:,}")

# ============================================================ 2. Pruning ===
print("\n" + "=" * 70)
print("STEP 2 — BUILD FULL GRAPH & PRUNE")
print("=" * 70)

# Build the full undirected weighted graph from the upper-triangle slice
print(f"\nBuilding undirected graph with {n_edges_before:,} edges ...")
G = nx.from_pandas_edgelist(
    upper,
    source="product_i",
    target="product_j",
    edge_attr="phi",
)
print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# 2a. Maximum spanning tree (by phi weight)
print(f"\nComputing maximum spanning tree (Kruskal) ...")
mst = nx.maximum_spanning_tree(G, weight="phi", algorithm="kruskal")
print(f"  MST edges: {mst.number_of_edges():,}")

# 2b. Add back all edges where phi >= threshold
high_phi_edges = [
    (u, v, d) for u, v, d in G.edges(data=True)
    if d["phi"] >= PHI_PRUNE_THRESH
]
print(f"  Edges with phi >= {PHI_PRUNE_THRESH}: {len(high_phi_edges):,}")

# Union: start from MST, add high-phi edges
pruned = nx.Graph(mst)
pruned.add_edges_from(high_phi_edges)
n_edges_after = pruned.number_of_edges()
print(f"\nEdge count AFTER pruning (MST ∪ phi≥{PHI_PRUNE_THRESH}): {n_edges_after:,}")
print(f"  Edges removed: {n_edges_before - n_edges_after:,}  "
      f"({(n_edges_before - n_edges_after) / n_edges_before * 100:.1f}% reduction)")

# ======================================= 3. Fixed-seed spring layout ===
print("\n" + "=" * 70)
print("STEP 3 — SPRING LAYOUT (seed={})".format(LAYOUT_SEED))
print("=" * 70)

print(f"\nComputing spring layout for {pruned.number_of_nodes():,} nodes ...")
pos = nx.spring_layout(pruned, seed=LAYOUT_SEED, weight="phi", k=0.3)
print(f"  Layout computed for {len(pos):,} nodes.")

# =========================================== 4. Build node DataFrame ===
print("\n" + "=" * 70)
print("STEP 4 — ATTACH METADATA & SAVE")
print("=" * 70)

# Node table: product, x, y, name, sector, pci
prod_meta = prod.set_index("product")[[  "name", "sector", "pci_latest"]].copy()
prod_meta.index = prod_meta.index.astype(str).str.strip().str.upper()

node_rows = []
for node, (x, y) in pos.items():
    meta = prod_meta.loc[node] if node in prod_meta.index else pd.Series({"name": None, "sector": None, "pci_latest": None})
    node_rows.append({
        "product":  node,
        "x":        float(x),
        "y":        float(y),
        "name":     meta["name"],
        "sector":   meta["sector"],
        "pci":      meta["pci_latest"],
    })

nodes_df = pd.DataFrame(node_rows).sort_values("product").reset_index(drop=True)
missing_meta = nodes_df["name"].isna().sum()
print(f"\nNode table: {len(nodes_df):,} rows")
print(f"  Nodes with missing name/sector: {missing_meta}")
print(f"  Sector distribution:\n{nodes_df['sector'].value_counts().head(10).to_string()}")

# Edge table
edge_rows = []
for u, v, d in pruned.edges(data=True):
    edge_rows.append({"product_i": u, "product_j": v, "phi": d["phi"]})
edges_df = pd.DataFrame(edge_rows).sort_values(["product_i", "product_j"]).reset_index(drop=True)

print(f"\nEdge table: {len(edges_df):,} rows")
print(f"  phi stats:\n{edges_df['phi'].describe().to_string()}")

# ======================================================== 5. Save outputs ===
nodes_df.to_parquet(OUT_NODES, index=False)
edges_df.to_parquet(OUT_EDGES, index=False)

# Also save a combined sentinel (network_A2010.parquet) with a 'table' tag column
nodes_tagged = nodes_df.copy()
nodes_tagged["table"] = "nodes"
edges_tagged = edges_df.copy()
edges_tagged["table"] = "edges"
combined = pd.concat([nodes_tagged, edges_tagged], ignore_index=True, sort=False)
combined.to_parquet(OUT_DIR / "network_A2010.parquet", index=False)

print(f"\nSaved:")
print(f"  Nodes → {OUT_NODES}")
print(f"  Edges → {OUT_EDGES}")
print(f"  Combined → {OUT_DIR / 'network_A2010.parquet'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Proximity file:              proximity_A2010.parquet")
print(f"  Products in products_v1:     {len(prod_products):,}")
print(f"  Edges BEFORE pruning:        {n_edges_before:,}")
print(f"  MST edges:                   {mst.number_of_edges():,}")
print(f"  High-phi (>=0.4) edges:      {len(high_phi_edges):,}")
print(f"  Edges AFTER pruning:         {n_edges_after:,}")
print(f"  Layout seed:                 {LAYOUT_SEED}")
print(f"  Nodes in output:             {len(nodes_df):,}")
print("=" * 70)
print("Done.")


"""
plot_product_space_2010.py  —  Day 6: Interactive Product Space Visualization
=============================================================================
Creates an interactive plotly network graph of the pruned product space
with India's 2010 comparative advantages highlighted.

Requirements:
  - network_A2010_nodes.parquet  (product, x, y, name, sector, pci)
  - network_A2010_edges.parquet  (product_i, product_j, phi)
  - trade_rca_v1.parquet         (for India m=1 products in 2010)
  - predictions_density.parquet  (optional P3 scores for India asof 2010)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------------------------------------------------ paths ---
ROOT      = Path(__file__).resolve().parents[1]
NODES     = ROOT / "outputs" / "network" / "network_A2010_nodes.parquet"
EDGES     = ROOT / "outputs" / "network" / "network_A2010_edges.parquet"
TRADE     = ROOT / "data"  / "processed" / "trade_rca_v1.parquet"
PRED_PATH = ROOT / "p2" / "data-p2" / "predictions" / "predictions_density.parquet"
OUT_HTML  = ROOT / "outputs" / "network" / "product_space_2010.html"

# ================================================================ 1. Load ===
print("Loading data ...")

nodes = pd.read_parquet(NODES)
edges = pd.read_parquet(EDGES)

# India's current products (m == 1 in 2010)
trade = pd.read_parquet(TRADE, columns=["country", "product", "year", "m"])
india_2010 = trade[(trade["country"] == "IND") & (trade["year"] == 2010)]
india_present = set(india_2010.loc[india_2010["m"] == 1, "product"])

print(f"  Nodes: {len(nodes):,}  |  Edges: {len(edges):,}")
print(f"  India products present in 2010 (m=1): {len(india_present):,}")

# P3 predictions (optional)
pred_scores = {}
if PRED_PATH.exists():
    pred = pd.read_parquet(PRED_PATH)
    ind_pred = pred[(pred["country"] == "IND") & (pred["asof_year"] == 2010)]
    pred_scores = dict(zip(ind_pred["product"], ind_pred["score"]))
    print(f"  P3 predictions loaded for {len(pred_scores)} India products (asof 2010)")
else:
    print("  No P3 predictions file found; hover will show name/sector/pci only.")

del trade  # free memory

# ============================================================ 2. Prepare ===
# Mark India's current products
nodes["india_present"] = nodes["product"].isin(india_present)

# Attach prediction scores
nodes["prob"] = nodes["product"].map(pred_scores)

# -- Node sizing: scale pci to a visual marker size --
# Map pci to [6, 22] range — low-complexity products are small, high are large
pci_min, pci_max = nodes["pci"].min(), nodes["pci"].max()
nodes["size"] = 6 + 16 * (nodes["pci"] - pci_min) / (pci_max - pci_min)
nodes["size"] = nodes["size"].fillna(8)

# -- Sector colour palette --
# 20 sectors need a rich qualitative palette
SECTOR_COLORS = {
    "Agriculture, live animals and plant products":                    "#2ca02c",
    "Arms and ammunition":                                            "#7f7f7f",
    "Basic metals and fabricated metal products":                      "#8c564b",
    "Chemicals and petrochemicals":                                    "#9467bd",
    "Electrical machinery, electronics and electrical equipment":      "#1f77b4",
    "Food and animal products, beverages, fats, oils and tobacco":     "#ff7f0e",
    "Footwear, headgear and related products":                         "#e377c2",
    "Furniture, toys, miscellaneous manufactures and special goods":   "#bcbd22",
    "Industrial machinery and mechanical equipment":                   "#17becf",
    "Leather, hides, furs and related products":                       "#d62728",
    "Medical devices, optical and precision instruments":              "#aec7e8",
    "Mining and non-fuel mineral products":                            "#c49c94",
    "Petroleum, coal and refined fuels":                               "#393b79",
    "Pharmaceuticals and medicinal products":                          "#e7969c",
    "Plastics and rubber products":                                    "#de9ed6",
    "Precious metals, gems and jewellery":                             "#ffd92f",
    "Stone, ceramics, glass, cement and related products":             "#c7c7c7",
    "Textiles, apparel and textile articles":                          "#dbdb8d",
    "Transport equipment":                                             "#9edae5",
    "Wood, pulp, paper and printing products":                         "#98df8a",
}
DEFAULT_COLOR = "#cccccc"
nodes["color"] = nodes["sector"].map(SECTOR_COLORS).fillna(DEFAULT_COLOR)

# -- Hover text --
hover_parts = []
for _, r in nodes.iterrows():
    text = (
        f"<b>{r['name'] or r['product']}</b><br>"
        f"Product: {r['product']}<br>"
        f"Sector: {r['sector'] or 'Unknown'}<br>"
        f"PCI: {r['pci']:.3f}"
    )
    if pd.notna(r["prob"]):
        text += f"<br>P(emergence): {r['prob']:.4f}"
    if r["india_present"]:
        text += "<br><b>🇮🇳 India exports this (2010)</b>"
    hover_parts.append(text)
nodes["hover"] = hover_parts

# ========================================================== 3. Build fig ===
print("Building figure ...")

fig = go.Figure()

# ------- 3a. Edges as thin grey lines -------
# Collect all edge line segments as coordinate lists with None separators
# (much faster than one trace per edge)
edge_x = []
edge_y = []

node_pos = dict(zip(nodes["product"], zip(nodes["x"], nodes["y"])))

for _, e in edges.iterrows():
    p_i, p_j = e["product_i"], e["product_j"]
    if p_i in node_pos and p_j in node_pos:
        x0, y0 = node_pos[p_i]
        x1, y1 = node_pos[p_j]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode="lines",
    line=dict(width=0.3, color="rgba(180,180,180,0.35)"),
    hoverinfo="skip",
    showlegend=False,
    name="edges",
))

# ------- 3b. Nodes: non-India products (per sector, for legend) -------
non_india = nodes[~nodes["india_present"]]
india     = nodes[nodes["india_present"]]

# Add one trace per sector for legend grouping (non-India)
for sector in sorted(nodes["sector"].dropna().unique()):
    subset = non_india[non_india["sector"] == sector]
    if len(subset) == 0:
        continue
    color = SECTOR_COLORS.get(sector, DEFAULT_COLOR)
    fig.add_trace(go.Scatter(
        x=subset["x"], y=subset["y"],
        mode="markers",
        marker=dict(
            size=subset["size"],
            color=color,
            line=dict(width=0.5, color="rgba(80,80,80,0.4)"),
            opacity=0.75,
        ),
        text=subset["hover"],
        hoverinfo="text",
        name=sector,
        legendgroup=sector,
    ))

# ------- 3c. India-present nodes — bold highlighted ring -------
# First: draw the same per-sector points with a bold dark border
for sector in sorted(india["sector"].dropna().unique()):
    subset = india[india["sector"] == sector]
    if len(subset) == 0:
        continue
    color = SECTOR_COLORS.get(sector, DEFAULT_COLOR)
    fig.add_trace(go.Scatter(
        x=subset["x"], y=subset["y"],
        mode="markers",
        marker=dict(
            size=subset["size"] + 3,        # slightly larger
            color=color,
            line=dict(width=2.5, color="#222222"),  # bold black border
            opacity=1.0,
        ),
        text=subset["hover"],
        hoverinfo="text",
        name=f"IND: {sector}",
        legendgroup=sector,
        showlegend=False,     # avoid legend clutter; see annotation below
    ))

# A single dummy trace for the India legend entry
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers",
    marker=dict(
        size=12, color="white",
        line=dict(width=2.5, color="#222222"),
    ),
    name="🇮🇳 India exports (2010)",
    showlegend=True,
))

# ------- 3d. Layout -------
fig.update_layout(
    title=dict(
        text="Product Space — India 2010<br>"
             "<sup>Node colour = sector · Node size = PCI · "
             "Bold border = India comparative advantage (m=1)</sup>",
        x=0.5,
        font=dict(size=18),
    ),
    showlegend=True,
    legend=dict(
        title="Sectors",
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#ccc",
        borderwidth=1,
        itemsizing="constant",
        x=1.01, y=1, xanchor="left",
    ),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=1400,
    height=900,
    margin=dict(l=20, r=300, t=80, b=20),
    hovermode="closest",
)

# ============================================================ 4. Save ===
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")

print(f"\nSaved interactive graph → {OUT_HTML}")
print(f"  Total traces: {len(fig.data)}")
print(f"  India-present nodes highlighted: {len(india)}")
print(f"  Non-India nodes: {len(non_india)}")
print("Done.")


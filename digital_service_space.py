
import os
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = "software_results"
OUTPUT_DIR = os.path.join(INPUT_DIR, "visuals")

os.makedirs(OUTPUT_DIR, exist_ok=True)

INDIA = "IN"

SERVICES = {
    "SF": "Insurance and Pension",
    "SG": "Financial Services",
    "SH": "Intellectual Property Charges",
    "SI1": "Telecommunications",
    "SI2": "Computer Services",
    "SI3": "Information Services",
    "SJ": "Other Business Services",
    "SK": "Personal, Cultural and Recreational"
}

# Only show edges with proximity above this threshold.
# Change to 0 if you want to display every connection.
PROXIMITY_THRESHOLD = 0.0


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading service data...")

proximity = pd.read_csv(
    os.path.join(INPUT_DIR, "services_proximity.csv")
)

rca = pd.read_csv(
    os.path.join(INPUT_DIR, "services_rca.csv")
)

exports = pd.read_csv(
    os.path.join(INPUT_DIR, "services_exports.csv")
)

for df in [proximity, rca, exports]:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

proximity["proximity"] = pd.to_numeric(
    proximity["proximity"], errors="coerce"
)

rca["rca"] = pd.to_numeric(rca["rca"], errors="coerce")
exports["value"] = pd.to_numeric(
    exports["value"], errors="coerce"
)

proximity = proximity.dropna(
    subset=["year", "proximity"]
)

rca = rca.dropna(subset=["year"])
exports = exports.dropna(subset=["year", "value"])

# India-only RCA and export data
india_rca = rca[
    (rca["reporter_code"] == INDIA) &
    (rca["indicator"].isin(SERVICES))
].copy()

india_exports = exports[
    (exports["reporter_code"] == INDIA) &
    (exports["indicator"].isin(SERVICES))
].copy()

years = sorted(
    set(proximity["year"].unique()) &
    set(india_rca["year"].unique()) &
    set(india_exports["year"].unique())
)

if not years:
    raise ValueError("No common years found across the input files.")

print("Available years:", years)


# ============================================================
# 2. CREATE A FIXED NETWORK LAYOUT
# ============================================================

# Average proximity across years determines node positions.
# This keeps positions fixed when the year changes.

average_proximity = (
    proximity.groupby(
        ["service_1", "service_2"]
    )["proximity"]
    .mean()
    .reset_index()
)

G = nx.Graph()

for code in SERVICES:
    G.add_node(code)

for _, row in average_proximity.iterrows():

    a = row["service_1"]
    b = row["service_2"]

    if a in SERVICES and b in SERVICES and a != b:
        G.add_edge(
            a,
            b,
            weight=float(row["proximity"])
        )

# Force-directed layout
positions = nx.spring_layout(
    G,
    seed=42,
    weight="weight",
    k=1.2,
    iterations=200
)


# ============================================================
# 3. CREATE INTERACTIVE NETWORK
# ============================================================

fig = go.Figure()

trace_groups = {}

for year in years:

    year = int(year)
    start_trace = len(fig.data)

    # -----------------------------------------
    # PROXIMITY EDGES
    # -----------------------------------------

    year_proximity = proximity[
        proximity["year"] == year
    ]

    for _, row in year_proximity.iterrows():

        a = row["service_1"]
        b = row["service_2"]
        weight = float(row["proximity"])

        if a == b:
            continue

        if a not in SERVICES or b not in SERVICES:
            continue

        if weight < PROXIMITY_THRESHOLD:
            continue

        x0, y0 = positions[a]
        x1, y1 = positions[b]

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=0.5 + weight * 5,
                    color=f"rgba(80, 130, 180,{0.15 + 0.75 * weight})"
                ),
                hovertemplate=(
                    f"{SERVICES[a]} ↔ {SERVICES[b]}"
                    f"<br>Proximity: {weight:.3f}"
                    "<extra></extra>"
                ),
                showlegend=False
            )
        )

    # -----------------------------------------
    # INDIA'S RCA AND EXPORT VALUES
    # -----------------------------------------

    year_rca = india_rca[
        india_rca["year"] == year
    ].set_index("indicator")

    year_exports = india_exports[
        india_exports["year"] == year
    ].set_index("indicator")

    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    node_customdata = []

    values = []

    for code in SERVICES:
        if code in year_exports.index:
            values.append(
                float(year_exports.loc[code, "value"])
            )
        else:
            values.append(0.0)

    max_export = max(values) if values else 1.0

    for idx, code in enumerate(SERVICES):

        x, y = positions[code]

        if code in year_rca.index:
            rca_value = float(
                year_rca.loc[code, "rca"]
            )
        else:
            rca_value = 0.0

        export_value = values[idx]

        has_advantage = rca_value >= 1

        node_x.append(x)
        node_y.append(y)

        node_text.append(code)

        # Node size based on India's export value
        size = (
            15 + 30 * np.sqrt(export_value / max_export)
            if max_export > 0 else 15
        )

        node_sizes.append(size)

        # Green = RCA >= 1; grey = RCA < 1
        node_colors.append(
            "#16a34a" if has_advantage else "#9ca3af"
        )

        node_customdata.append([
            SERVICES[code],
            rca_value,
            export_value,
            "Yes" if has_advantage else "No"
        ])

    # -----------------------------------------
    # SERVICE NODES
    # -----------------------------------------

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=12),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(
                    color="black",
                    width=1
                ),
                opacity=0.95
            ),
            customdata=node_customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>RCA: %{customdata[1]:.3f}"
                "<br>Exports: %{customdata[2]:,.2f} million USD"
                "<br>Comparative advantage: %{customdata[3]}"
                "<extra></extra>"
            ),
            showlegend=False
        )
    )

    end_trace = len(fig.data)

    trace_groups[year] = list(
        range(start_trace, end_trace)
    )


# ============================================================
# 4. ADD YEAR SELECTOR
# ============================================================

total_traces = len(fig.data)

buttons = []

for year in years:

    visible = [False] * total_traces

    for trace_idx in trace_groups[int(year)]:
        visible[trace_idx] = True

    buttons.append(
        dict(
            label=str(int(year)),
            method="update",
            args=[
                {"visible": visible},
                {
                    "title": (
                        f"India Digital Services Space ({int(year)})"
                    )
                }
            ]
        )
    )

# Initially display the first year
initial_year = int(years[0])

initial_visibility = [False] * total_traces

for idx in trace_groups[initial_year]:
    initial_visibility[idx] = True

for i in range(total_traces):
    fig.data[i].visible = initial_visibility[i]


# ============================================================
# 5. FINAL LAYOUT
# ============================================================

fig.update_layout(
    title=f"India Digital Services Space ({initial_year})",

    template="plotly_white",

    width=1150,
    height=800,

    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.02,
            y=1.12,
            xanchor="left",
            yanchor="top"
        )
    ],

    annotations=[
        dict(
            text=(
                "<b>Green:</b> RCA ≥ 1 &nbsp;&nbsp; "
                "<b>Grey:</b> RCA < 1 &nbsp;&nbsp; "
                "<b>Node size:</b> India's exports &nbsp;&nbsp; "
                "<b>Edge thickness:</b> Proximity"
            ),
            x=0.5,
            y=1.06,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12)
        )
    ],

    xaxis=dict(
        visible=False,
        showgrid=False,
        zeroline=False
    ),

    yaxis=dict(
        visible=False,
        showgrid=False,
        zeroline=False
    ),

    hovermode="closest",

    margin=dict(
        l=40,
        r=40,
        t=130,
        b=40
    )
)

# ============================================================
# 6. SAVE INTERACTIVE HTML
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "india_digital_services_space.html"
)

fig.write_html(
    output_file,
    include_plotlyjs=True
)

print("\n====================================")
print("DIGITAL SERVICES SPACE GENERATED")
print("====================================")
print("File:", output_file)
print("Years:", years)
print("Service categories:", len(SERVICES))
print("Open the HTML file in your browser.")

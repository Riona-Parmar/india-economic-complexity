"""
compute_geopolitical_exposure.py — Phase 3 Geopolitical Exposure Feature Construction
====================================================================================
1. Fetches country-level Geopolitical Risk (GPR / AI-GPR) index data from
   https://matteoiacoviello.com/gpr_country.htm and https://matteoiacoviello.com/ai_gpr.html
2. Loads global trade data (trade_rca_v1.parquet and country_year_sector_trade.csv)
   to calculate world import demand for each of the 1,243 products.
3. For every single product, identifies its top 30 importing partner countries.
4. Calculates product-level geopolitical risk exposure by weighting the import demand
   shares of the top 30 importing partners by their country GPR scores.
5. Saves the final product-level features to outputs/geopolitical_features.parquet.
6. Prints a preview of the top 5 products with the highest geopolitical risk exposure.
"""

from pathlib import Path
import io
import sys
import numpy as np
import pandas as pd
import requests
import pycountry

# ------------------------------------------------------------------ paths ---
ROOT        = Path(__file__).resolve().parents[1]
DATA_DIR    = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim"
OUT_DIR     = ROOT / "outputs"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRADE_PATH    = DATA_DIR / "trade_rca_v1.parquet"
SECTOR_PATH   = DATA_DIR / "country_year_sector_trade.csv"
PRODUCTS_PATH = DATA_DIR / "products_v1.parquet"
OUTPUT_PARQUET = OUT_DIR / "geopolitical_features.parquet"

# Cached GPR download locations
GPR_AI_CSV    = INTERIM_DIR / "ai_gpr_country_monthly.csv"
GPR_XLS       = INTERIM_DIR / "data_gpr_export.xls"

# ------------------------------------------------------------------ step 1 ---
print("=" * 70)
print("STEP 1: Fetching Geopolitical Risk (GPR) Index Data")
print("=" * 70)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 1a. AI-GPR (200 countries)
URL_AI_GPR = "https://www.matteoiacoviello.com/ai_gpr_files/ai_gpr_country_monthly.csv"
if not GPR_AI_CSV.exists() or GPR_AI_CSV.stat().st_size == 0:
    print(f"Downloading AI-GPR country data from {URL_AI_GPR} ...")
    resp = requests.get(URL_AI_GPR, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    GPR_AI_CSV.write_bytes(resp.content)
    print(f"Saved: {GPR_AI_CSV} ({len(resp.content)/1024:.1f} KB)")
else:
    print(f"Using cached AI-GPR country data: {GPR_AI_CSV}")

df_ai_gpr = pd.read_csv(GPR_AI_CSV)
df_ai_gpr["Date"] = pd.to_datetime(df_ai_gpr["Date"])
print(f"AI-GPR time range: {df_ai_gpr['Date'].min().strftime('%Y-%m')} to {df_ai_gpr['Date'].max().strftime('%Y-%m')}")

# 1b. Benchmark GPR (44 countries)
URL_BENCH_GPR = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
if not GPR_XLS.exists() or GPR_XLS.stat().st_size == 0:
    print(f"Downloading benchmark GPR data from {URL_BENCH_GPR} ...")
    try:
        resp_xls = requests.get(URL_BENCH_GPR, headers=HEADERS, timeout=30)
        resp_xls.raise_for_status()
        GPR_XLS.write_bytes(resp_xls.content)
        print(f"Saved: {GPR_XLS} ({len(resp_xls.content)/1024:.1f} KB)")
    except Exception as e:
        print(f"Warning: Could not download {URL_BENCH_GPR}: {e}")
else:
    print(f"Using cached benchmark GPR data: {GPR_XLS}")

# Compute recent 5-year average GPR (2020-2024) matching project's forecast horizon
sub_ai = df_ai_gpr[df_ai_gpr["Date"].dt.year.between(2020, 2024)]
gpr_means = sub_ai.mean(numeric_only=True)

# Map AI-GPR country names to ISO3
country_aliases = {
    "USA": "USA", "UK": "GBR", "Russia": "RUS", "Vietnam": "VNM", "Iran": "IRN",
    "Syria": "SYR", "Palestine": "PSE", "North Korea": "PRK", "South Korea": "KOR",
    "DRC": "COD", "Congo": "COG", "Czechia": "CZE", "Taiwan": "TWN", "Laos": "LAO",
    "Moldova": "MDA", "Tanzania": "TZA", "Bolivia": "BOL", "Venezuela": "VEN",
    "Ivory Coast": "CIV", "Brunei": "BRN", "Cape Verde": "CPV", "Micronesia": "FSM",
    "East Timor": "TLS", "Kosovo": "XKX", "Vatican": "VAT", "Burma": "MMR",
    "Turkey": "TUR", "Swaziland": "SWZ", "St. Lucia": "LCA", "Netherlands": "NLD",
    "Hong Kong": "HKG", "UAE": "ARE", "Singapore": "SGP"
}

iso_to_gpr = {}
for col in df_ai_gpr.columns:
    if col.endswith("_all"):
        c_name = col[:-4]
        iso = country_aliases.get(c_name)
        if not iso:
            try:
                matches = pycountry.countries.search_fuzzy(c_name)
                if matches:
                    iso = matches[0].alpha_3
            except Exception:
                pass
        if iso:
            iso_to_gpr[iso] = float(gpr_means[col])

# Fallback for Hong Kong / territories if not explicit
if "CHN" in iso_to_gpr and "HKG" not in iso_to_gpr:
    iso_to_gpr["HKG"] = iso_to_gpr["CHN"]

print(f"Successfully mapped {len(iso_to_gpr)} country GPR scores.")
print("Sample country GPR scores (2020-2024 average):")
for sample_iso in ["USA", "RUS", "CHN", "UKR", "ISR", "DEU", "IND", "JPN", "GBR", "FRA"]:
    if sample_iso in iso_to_gpr:
        print(f"  {sample_iso}: {iso_to_gpr[sample_iso]:.2f}")

# ------------------------------------------------------------------ step 2 ---
print("\n" + "=" * 70)
print("STEP 2: Loading Global Trade Data and Computing Product World Demand")
print("=" * 70)

print(f"Loading {TRADE_PATH.name} ...")
trade = pd.read_parquet(TRADE_PATH, columns=["country", "product", "year", "export_value"])
trade_2024 = trade[trade["year"] == 2024].copy()

# World export value of product p = World import demand of product p
world_demand = trade_2024.groupby("product")["export_value"].sum().to_dict()
total_world_demand = sum(world_demand.values())
print(f"Total global merchandise import demand (2024): ${total_world_demand:,.2f} (~${total_world_demand/1e12:.2f} Trillion)")

print(f"\nLoading {SECTOR_PATH.name} ...")
sec_trade = pd.read_csv(SECTOR_PATH)
sec_2024 = sec_trade[sec_trade["year"] == 2024].copy()
sec_tot_imp = sec_2024.groupby("sector_name")["import_value"].sum().to_dict()

# Matrix of country import values by sector
sec_imp_matrix = sec_2024.pivot(
    index="country_iso3_code", columns="sector_name", values="import_value"
).fillna(0.0)

# Matrix of country exports by product
prod_exp_matrix = trade_2024.pivot(
    index="country", columns="product", values="export_value"
).fillna(0.0)

print(f"\nLoading {PRODUCTS_PATH.name} ...")
products = pd.read_parquet(PRODUCTS_PATH)
print(f"Loaded {len(products):,} products across {products['sector'].nunique()} sectors.")

# ------------------------------------------------------------------ step 3 & 4 ---
print("\n" + "=" * 70)
print("STEP 3 & 4: Identifying Top 30 Importing Partners and Weighting GPR Exposure")
print("=" * 70)

results = []

for _, row in products.iterrows():
    p = row["product"]
    p_name = row["name"]
    sec = row["sector"]
    pci = row["pci_latest"]
    
    W_p = float(world_demand.get(p, 0.0))
    M_s = float(sec_tot_imp.get(sec, 1.0))
    
    # Base sector imports by country
    if sec in sec_imp_matrix.columns:
        base_imp = sec_imp_matrix[sec]
    else:
        base_imp = pd.Series(0.0, index=sec_imp_matrix.index)
        
    # Country exports of product p
    if p in prod_exp_matrix.columns:
        c_exp = prod_exp_matrix[p].reindex(base_imp.index, fill_value=0.0)
    else:
        c_exp = pd.Series(0.0, index=base_imp.index)
        
    # Net product import demand by partner country
    # Countries that export product p supply part of their own demand;
    # countries with zero exports must import 100% of their sector share.
    export_intensity = (c_exp / max(1.0, W_p)).clip(0.0, 1.0)
    imp_demand = np.maximum(0.0, base_imp * (W_p / max(1.0, M_s)) * (1.0 - export_intensity))
    
    # If imp_demand is zero (e.g. W_p == 0), fallback to base sector imports
    if imp_demand.sum() == 0.0 and base_imp.sum() > 0:
        imp_demand = base_imp * (W_p / max(1.0, M_s))
        
    # Top 30 importing partner countries for this product
    top30 = imp_demand.sort_values(ascending=False).head(30)
    top30_sum = float(top30.sum())
    
    if top30_sum > 0:
        top30_shares = top30 / top30_sum
        top30_gpr_scores = pd.Series([iso_to_gpr.get(c, 0.0) for c in top30.index], index=top30.index)
        weighted_gpr = float((top30_shares * top30_gpr_scores).sum())
        top30_world_share = float(top30_sum / W_p) if W_p > 0 else 0.0
        top1_country = str(top30.index[0])
        top1_share = float(top30_shares.iloc[0])
        top1_gpr = float(iso_to_gpr.get(top1_country, 0.0))
    else:
        weighted_gpr = 0.0
        top30_world_share = 0.0
        top1_country = "N/A"
        top1_share = 0.0
        top1_gpr = 0.0
        
    # Total dollar value exposure to geopolitical risk
    val_exposure = float(W_p * weighted_gpr)
    
    results.append({
        "product": p,
        "name": p_name,
        "sector": sec,
        "pci": pci,
        "world_import_demand": W_p,
        "top30_import_demand": top30_sum,
        "top30_import_share": top30_world_share,
        "geopolitical_exposure": weighted_gpr,
        "geopolitical_value_exposure": val_exposure,
        "top_importer_country": top1_country,
        "top_importer_share": top1_share,
        "top_importer_gpr": top1_gpr,
    })

features_df = pd.DataFrame(results)

# ------------------------------------------------------------------ step 5 ---
print("\n" + "=" * 70)
print("STEP 5: Saving Geopolitical Features to Parquet")
print("=" * 70)

features_df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"Saved: {OUTPUT_PARQUET}")
print(f"File size: {OUTPUT_PARQUET.stat().st_size / 1024:.1f} KB")
print(f"Rows: {len(features_df):,}  |  Columns: {features_df.columns.tolist()}")

# Feature statistics
print("\nSummary Statistics of Geopolitical Exposure:")
print(features_df["geopolitical_exposure"].describe().to_string())

# ------------------------------------------------------------------ step 6 ---
print("\n" + "=" * 70)
print("STEP 6: Top 5 Products with Highest Geopolitical Risk Exposure")
print("=" * 70)

# Sort by geopolitical exposure rate
top5_rate = features_df.sort_values("geopolitical_exposure", ascending=False).head(5)

print("\n--- TOP 5 PRODUCTS BY GEOPOLITICAL RISK EXPOSURE (GPR Index) ---")
for idx, (_, r) in enumerate(top5_rate.iterrows(), 1):
    print(f"\n{idx}. Product {r['product']} — {r['name']}")
    print(f"   Sector:              {r['sector']}")
    print(f"   PCI:                 {r['pci']}")
    print(f"   Geopolitical Risk:   {r['geopolitical_exposure']:.4f}")
    print(f"   World Import Demand: ${r['world_import_demand']:,.2f} (~${r['world_import_demand']/1e9:.2f}B)")
    print(f"   Top Importer:        {r['top_importer_country']} (Share: {r['top_importer_share']:.2%}, GPR: {r['top_importer_gpr']:.2f})")

# Also show top 5 by total dollar value at risk
print("\n" + "-" * 70)
print("--- TOP 5 PRODUCTS BY DOLLAR-WEIGHTED GEOPOLITICAL VALUE EXPOSURE ---")
top5_val = features_df.sort_values("geopolitical_value_exposure", ascending=False).head(5)
for idx, (_, r) in enumerate(top5_val.iterrows(), 1):
    print(f"\n{idx}. Product {r['product']} — {r['name']}")
    print(f"   Sector:              {r['sector']}")
    print(f"   World Demand:        ${r['world_import_demand']:,.2f} (~${r['world_import_demand']/1e9:.2f}B)")
    print(f"   GPR Exposure Score:  {r['geopolitical_exposure']:.4f}")
    print(f"   Value at Risk:       {r['geopolitical_value_exposure']:,.2f} (~${r['geopolitical_value_exposure']/1e12:.2f}T)")

print("\n" + "=" * 70)
print("GEOPOLITICAL EXPOSURE TASK COMPLETED SUCCESSFULLY!")
print("=" * 70)


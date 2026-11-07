"""
verify_india_trade_and_labels.py

1. Calculates India's total exports for the years 2010, 2015, and 2020 from trade_rca_v1.parquet.
2. Recounts India's positive emergence labels (where rca3 >= 1.0 at as-of year + 5)
   and compares them directly against the numbers recorded in data_report.md.
"""

from pathlib import Path
import re
import sys
import pandas as pd

def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    parquet_path = data_dir / "processed" / "trade_rca_v1.parquet"
    report_path = root / "outputs" / "reports" / "data_report.md"
    
    print(f"Parquet file path: {parquet_path}")
    print(f"Data report path:  {report_path}")
    print("-" * 70)
    
    # -------------------------------------------------------------
    # 1. Load trade_rca_v1.parquet
    # -------------------------------------------------------------
    try:
        df = pd.read_parquet(
            parquet_path,
            columns=["country", "product", "year", "export_value", "rca3"],
        )
        print(f"Successfully loaded '{parquet_path.name}' with {len(df):,} rows.")
    except Exception as e:
        print(f"ERROR reading parquet file: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Filter for India (country code 'IND')
    india_df = df[df["country"] == "IND"].copy()
    print(f"Total rows for India (IND): {len(india_df):,}")
    print("-" * 70)
    
    # -------------------------------------------------------------
    # 2. Calculate India's total exports for 2010, 2015, and 2020
    # -------------------------------------------------------------
    target_export_years = [2010, 2015, 2020]
    export_totals = {}
    
    print("\n--- India's Total Exports ---")
    for y in target_export_years:
        val = india_df[india_df["year"] == y]["export_value"].sum()
        export_totals[y] = val
        print(f"Year {y}: ${val:,.2f} ({val:.4e}) (~${val / 1e9:.2f} Billion)")
    print("-" * 70)

    # -------------------------------------------------------------
    # 3. Recount India's Positive Labels (as-of year + 5, rca3 >= 1.0)
    # -------------------------------------------------------------
    print("\n--- Recounting India's Positive Labels ---")
    
    # Pivot rca3 by product and year
    wide_rca3 = india_df.pivot(index="product", columns="year", values="rca3")
    first_year = int(wide_rca3.columns.min())  # 1995
    windows = [2000, 2005, 2010, 2015]
    
    # Strict historical eligibility rule (as defined in data_report.md & build_products_and_labels_v1.py):
    # rca3 < 0.25 across all historical years from first_year (1995) to as-of year,
    # and rca3 >= 1.0 at target_year (as-of + 5).
    recounted_strict = {}
    eligible_counts = {}
    
    # Single-year baseline eligibility (rca3 < 0.25 at as-of year only):
    recounted_singleyear = {}
    
    for asof in windows:
        target = asof + 5
        # Strict eligibility
        hist_years = list(range(first_year, asof + 1))
        hist_series = wide_rca3[hist_years]
        strict_eligible = hist_series.lt(0.25).all(axis=1)
        strict_positives = (wide_rca3.loc[strict_eligible, target] >= 1.0).sum()
        
        eligible_counts[asof] = strict_eligible.sum()
        recounted_strict[asof] = strict_positives
        
        # Single-year eligibility
        sy_eligible = wide_rca3[asof] < 0.25
        sy_positives = (wide_rca3.loc[sy_eligible, target] >= 1.0).sum()
        recounted_singleyear[asof] = sy_positives

    # -------------------------------------------------------------
    # 4. Compare with numbers in data_report.md
    # -------------------------------------------------------------
    print("\n--- Comparison with data_report.md ---")
    report_numbers = {}
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
        # Match table rows: | 2000 | 2005 | 392 | 12 |
        matches = re.findall(r"\|\s*(\d{4})\s*\|\s*(\d{4})\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|", content)
        for asof_s, target_s, elig_s, pos_s in matches:
            asof_int = int(asof_s)
            if asof_int in windows:
                report_numbers[asof_int] = {
                    "target_year": int(target_s),
                    "eligible": int(elig_s.replace(",", "")),
                    "positives": int(pos_s.replace(",", "")),
                }

    header = f"{'As-of':<8} | {'Target':<8} | {'Eligible (Recount)':<20} | {'Recount Positives':<18} | {'Report Positives':<18} | {'Match?':<8}"
    print(header)
    print("-" * len(header))
    
    all_match = True
    for asof in windows:
        target = asof + 5
        rec_pos = recounted_strict[asof]
        rec_elig = eligible_counts[asof]
        rep_pos = report_numbers.get(asof, {}).get("positives", "N/A")
        match_str = "YES" if rec_pos == rep_pos else "NO"
        if rec_pos != rep_pos:
            all_match = False
        print(f"{asof:<8} | {target:<8} | {rec_elig:<20} | {rec_pos:<18} | {str(rep_pos):<18} | {match_str:<8}")
        
    print("-" * len(header))
    total_recounted = sum(recounted_strict.values())
    total_reported = sum(v["positives"] for v in report_numbers.values()) if report_numbers else "N/A"
    print(f"Total across all windows: Recounted = {total_recounted}, Reported in data_report.md = {total_reported} (Match: {total_recounted == total_reported})")

    print("\nNote on single-year eligibility (rca3 < 0.25 at as-of year only, not strict 1995-to-asof):")
    for asof in windows:
        print(f"  {asof} -> {asof + 5}: {recounted_singleyear[asof]} positive transitions")

if __name__ == "__main__":
    main()


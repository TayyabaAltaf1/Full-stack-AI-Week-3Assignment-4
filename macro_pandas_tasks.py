# macro_pandas_tasks.py
# Put macro_monthly.csv in same folder OR set CSV variable to a raw GitHub URL.

import pandas as pd
import numpy as np
import re

# --- CONFIG: change to raw URL if you prefer (uncomment) ---
CSV = "macro_monthly.csv"
CSV = "https://raw.githubusercontent.com/ShahzadSarwar10/Fullstack-WITH-AI-B-3-SAT-SUN-6Months-Explorer/refs/heads/main/DataSetForPractice/macro_monthly.csv"

# Utility: robust column finder (ignores case, spaces, punctuation)
def normalize_name(s):
    if s is None:
        return ""
    return re.sub(r'[^a-z0-9]', '', s.strip().lower())

def find_col(df, want):
    want_n = normalize_name(want)
    for c in df.columns:
        if normalize_name(c) == want_n:
            return c
    # fallback: contains match
    for c in df.columns:
        if want_n in normalize_name(c):
            return c
    return None

def safe_load(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV: {csv_path}  (rows={len(df)}, cols={len(df.columns)})")
        return df
    except Exception as e:
        raise SystemExit(f"Failed to load CSV '{csv_path}': {e}\nMake sure file exists in folder or set correct raw URL.")

def show_section(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def short_analysis(msg):
    print(">> Analysis:", msg, "\n")

def main():
    df = safe_load(CSV)

    # TASK 1: Print DataFrame (auto index)
    show_section("Task 1: DataFrame (auto index) — print first 20 rows & shape")
    print(df.head(20).to_string(index=True))
    print("\nShape:", df.shape)
    short_analysis("Default pandas index is RangeIndex starting at 0 unless CSV had an index column.")

    # Map required columns to actual names (robust)
    cols = {
        "Year": find_col(df, "Year"),
        "Industrial_Production": find_col(df, "Industrial_Production"),
        "Manufacturers_New_Orders_Durable_Goods": find_col(df, "Manufacturers_New_Orders: Durable Goods"),
        "Unemployment_Rate": find_col(df, "Unemployment_Rate"),
        "Retail_Sales": find_col(df, "Retail_Sales"),
        "Personal_Consumption_Expenditures": find_col(df, "Personal_Consumption_Expenditures"),
        "Consumer_Price_Index": find_col(df, "Consumer_Price Index"),
        "All_Employees": find_col(df, "All_Employees"),
        "National_Home_Price_Index": find_col(df, "National_Home_Price_Index"),
        "zip": find_col(df, "zip_code")  # example if present
    }
    show_section("Detected column mapping (requested -> found)")
    for k,v in cols.items():
        print(f"{k:45} -> {v}")
    print()

    # TASK 2: .info(), .dtypes, .describe(), .shape
    show_section("Task 2: DataFrame info / dtypes / describe / shape")
    print("df.info():")
    df.info()
    print("\n.dtypes:\n", df.dtypes)
    print("\n.describe() (numeric):\n", df.describe().to_string())
    print("\n.shape:", df.shape)
    short_analysis("info() shows non-null counts and types. describe() summarizes numeric columns (count, mean, std, min, max, percentiles).")

    # TASK 3: explore .to_string() with examples
    show_section("Task 3: DataFrame.to_string() examples")
    sample = df.head(12)
    print("Default sample.to_string():\n", sample.to_string())
    # subset columns
    subset_cols = [cols[x] for x in ("Industrial_Production","Manufacturers_New_Orders_Durable_Goods") if cols.get(x)]
    print("\nSubset to_string (no index):")
    if subset_cols:
        print(sample.to_string(columns=subset_cols, index=False))
    else:
        print("Subset columns not found to demo.")
    print("\nWith na_rep and float_format:")
    print(sample.to_string(na_rep="(missing)", float_format=lambda x: f"{x:.3f}" if pd.notna(x) else x, line_width=100))
    print("\nWith max_rows=5, max_cols=5:")
    print(sample.to_string(max_rows=5, max_cols=5, show_dimensions=True))
    short_analysis("Use to_string parameters to control output formatting for printing or writing to text buffer/file.")

    # TASK 4: top 4 rows
    show_section("Task 4: Top 4 rows (head(4))")
    print(df.head(4).to_string())
    short_analysis("head(4) returns first 4 rows based on current index order.")

    # TASK 5: bottom 4 rows
    show_section("Task 5: Bottom 4 rows (tail(4))")
    print(df.tail(4).to_string())
    short_analysis("tail(4) returns last 4 rows.")

    # TASK 6: access columns Industrial_Production and Manufacturers_New_Orders: Durable Goods
    show_section("Task 6: Print full columns (Industrial_Production & Manufacturers_New_Orders: Durable Goods)")
    if cols["Industrial_Production"]:
        print("Industrial_Production column values (first 20):")
        print(df[cols["Industrial_Production"]].head(20).to_string())
    else:
        print("Industrial_Production column not found.")
    if cols["Manufacturers_New_Orders_Durable_Goods"]:
        print("\nManufacturers_New_Orders: Durable Goods (first 20):")
        print(df[cols["Manufacturers_New_Orders_Durable_Goods"]].head(20).to_string())
    else:
        print("Manufacturers_New_Orders: Durable Goods column not found.")
    short_analysis("Access columns by df['colname'] — prints whole column (we show first 20 here).")

    # TASK 7: access multiple columns
    show_section("Task 7: Multiple columns (Industrial_Production & Manufacturers_New_Orders)")
    if subset_cols:
        print(df.loc[:, subset_cols].head(10).to_string())
        short_analysis("Using df.loc[:, [col1, col2]] returns those columns for all rows.")
    else:
        print("Required columns for this task not found.")

    # TASK 8: Selecting single row using .loc with auto index value 3
    show_section("Task 8: .loc[3] (label-based - index label = 3)")
    if 3 in df.index:
        row3 = df.loc[3]
        print(row3.to_string())
        short_analysis(".loc uses index labels; with default RangeIndex 0..n-1 label 3 is the 4th row.")
    else:
        print("Index label 3 not present (index may be different). Use df.reset_index() to reindex if needed.")

    # TASK 9: .loc multiple rows 3,5,7
    show_section("Task 9: .loc[[3,5,7]]")
    wanted = [i for i in [3,5,7] if i in df.index]
    if wanted:
        print(df.loc[wanted].to_string())
        short_analysis("Returned rows correspond to labels 3,5,7 (inclusive).")
    else:
        print("None of requested labels present in index.")

    # TASK 10: .loc slice rows 5 to 15 (label-based, inclusive)
    show_section("Task 10: .loc[5:15] (label slice inclusive)")
    try:
        slice_5_15 = df.loc[5:15]
        print(slice_5_15.to_string())
        short_analysis(".loc label-slice includes both ends if labels exist.")
    except Exception as e:
        print("Error in loc slice:", e)

    # TASK 11: Conditional .loc Year in (1993,1994,1997) and Unemployment_Rate >= 1
    show_section("Task 11: Year in (1993,1994,1997) & Unemployment_Rate >= 1")
    year_col = cols.get("Year")
    unemp_col = cols.get("Unemployment_Rate")
    if year_col and unemp_col:
        cond = df[year_col].isin([1993,1994,1997]) & (df[unemp_col] >= 1)
        res = df.loc[cond]
        print(res.to_string() if not res.empty else "No matching rows.")
        short_analysis(f"Found {len(res)} rows matching the criterion.")
    else:
        print("Year or Unemployment_Rate column not found.")

    # TASK 12: .loc single row index 9 selecting specific columns
    show_section("Task 12: .loc[9, [Industrial_Production,Retail_Sales,Manufacturers_New_Orders: Durable Goods,Personal_Consumption_Expenditures]]")
    col_list = [cols.get("Industrial_Production"), cols.get("Retail_Sales"), cols.get("Manufacturers_New_Orders_Durable_Goods"), cols.get("Personal_Consumption_Expenditures")]
    col_list = [c for c in col_list if c]
    if 9 in df.index and col_list:
        try:
            print(df.loc[9, col_list].to_string())
        except Exception as e:
            print("Error selecting .loc[9, cols]:", e)
        short_analysis(".loc with scalar row and list of columns returns a Series with selected columns.")
    else:
        print("Index 9 or one of required columns not available.")

    # TASK 13: Selecting rows where Industrial_Production <= 0.5
    show_section("Task 13: Rows with Industrial_Production <= 0.5")
    if cols.get("Industrial_Production"):
        cond = df[cols["Industrial_Production"]] <= 0.5
        print(df.loc[cond].to_string() if not df.loc[cond].empty else "No rows with Industrial_Production <= 0.5")
        short_analysis("Numeric comparison selects rows where column value <= 0.5.")
    else:
        print("Industrial_Production column not found.")

    # TASK 14: Combined .loc: Industrial_Production <=0.5 AND Consumer_Price_Index > 0.2
    show_section("Task 14: Industrial_Production <=0.5 AND Consumer_Price_Index > 0.2")
    c1 = cols.get("Industrial_Production"); c2 = cols.get("Consumer_Price_Index")
    if c1 and c2:
        cond = (df[c1] <= 0.5) & (df[c2] > 0.2)
        res = df.loc[cond]
        print(res.to_string() if not res.empty else "No rows matching both conditions.")
        short_analysis(f"Rows found: {len(res)}")
    else:
        print("Required columns not found.")

    # TASK 15: .iloc select 4th row -> iloc[3]
    show_section("Task 15: .iloc select 4th row (iloc[3])")
    try:
        print(df.iloc[3].to_string())
        short_analysis("iloc uses 0-based integer positions; 4th row = iloc[3].")
    except Exception as e:
        print("Cannot select 4th row via iloc:", e)

    # TASK 16: .iloc multiple rows: 2nd,7th,8th,36th,9th => positions [1,6,7,35,8] (guard for range)
    show_section("Task 16: .iloc select rows positions [1,6,7,35,8] (2nd,7th,8th,36th,9th)")
    positions = [1,6,7,35,8]
    positions = [p for p in positions if p < len(df)]
    if positions:
        print(df.iloc[positions].to_string())
        short_analysis(f"Returned {len(positions)} rows using iloc positions.")
    else:
        print("Requested iloc positions out of range for this DataFrame.")

    # TASK 17: .iloc slice from 10th to 23rd row -> positions 9:23 (iloc[9:23])
    show_section("Task 17: .iloc slice 10th to 23rd -> iloc[9:23]")
    try:
        print(df.iloc[9:23].to_string())
        short_analysis(f"Returned rows {9} to {22} (0-based iloc slice end exclusive).")
    except Exception as e:
        print("Error in iloc slice:", e)

    # TASK 18: .iloc select single column 5th column -> iloc[:,4]
    show_section("Task 18: .iloc select 5th column (position 4)")
    if df.shape[1] > 4:
        print(df.iloc[:,4].head(20).to_string())
        short_analysis("This returns the entire 5th column (by position).")
    else:
        print("DataFrame has fewer than 5 columns.")

    # TASK 19: .iloc multiple columns 2nd,3rd,8th -> positions [1,2,7]
    show_section("Task 19: .iloc select columns positions [1,2,7]")
    cols_pos = [1,2,7]
    cols_pos = [p for p in cols_pos if p < df.shape[1]]
    if cols_pos:
        print(df.iloc[:, cols_pos].head(10).to_string())
        short_analysis("Selected multiple columns by integer positions.")
    else:
        print("Requested column positions out of range.")

    # TASK 20: .iloc slice columns 2nd to 8th -> iloc[:,1:8]
    show_section("Task 20: .iloc columns slice 2nd to 8th -> iloc[:,1:8]")
    try:
        print(df.iloc[:, 1:8].head(10).to_string())
    except Exception as e:
        print("Error in column slice:", e)

    # TASK 21: Combined .iloc select rows [4,5,7,25] -> positions [3,4,6,24] and columns [3rd,5th,7th] -> [2,4,6]
    show_section("Task 21: Combined iloc rows positions [3,4,6,24] and cols [2,4,6]")
    rows_pos = [p for p in [3,4,6,24] if p < len(df)]
    cols_pos = [p for p in [2,4,6] if p < df.shape[1]]
    if rows_pos and cols_pos:
        print(df.iloc[rows_pos, cols_pos].to_string())
        short_analysis("Combined row & column selection using integer positions.")
    else:
        print("Requested rows or columns out of range.")

    # TASK 22: Combined .iloc select range rows [3,34] and columns 3rd to 6th -> iloc[[2,33], 2:6]
    show_section("Task 22: Combined iloc rows 3 & 34 (positions 2,33) and cols 3rd to 6th (2:6)")
    rows_pick = [p for p in [2,33] if p < len(df)]
    if rows_pick and df.shape[1] >= 6:
        print(df.iloc[rows_pick, 2:6].to_string())
    else:
        print("Requested rows/columns out of range for this DataFrame.")

    # TASK 23: Add a new row (append)
    show_section("Task 23: Add a new row (append) and show df tail")
    new_row = {}
    for c in df.columns:
        # put example values or NaN depending on type
        if np.issubdtype(df[c].dtype, np.number):
            new_row[c] = 0
        else:
            new_row[c] = "NEW"
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    print("Appended new row. New shape:", df.shape)
    print(df.tail(3).to_string())
    short_analysis("Appending added a new row with default/example values. Index reset to RangeIndex.")

    # TASK 24: delete row with index 4 (label 4)
    show_section("Task 24: Delete row with index label 4 (if present)")
    if 4 in df.index:
        df = df.drop(index=4).reset_index(drop=True)
        print("Deleted label 4. New shape:", df.shape)
    else:
        print("Index label 4 not present; skipped deletion.")
    short_analysis("After drop we reset index to keep it sequential.")

    # TASK 25: delete rows with index from 5 to 9 (labels 5..9)
    show_section("Task 25: Delete rows labels 5 to 9 (if present)")
    labels = [i for i in range(5,10) if i in df.index]
    if labels:
        df = df.drop(index=labels).reset_index(drop=True)
        print(f"Dropped labels {labels}. New shape: {df.shape}")
    else:
        print("No labels 5..9 present; no rows dropped.")

    # TASK 26: Delete 'All_Employees' column
    show_section("Task 26: Delete column 'All_Employees' if present")
    all_emp = find_col(df, "All_Employees")
    if all_emp and all_emp in df.columns:
        df = df.drop(columns=[all_emp])
        print(f"Dropped column {all_emp}. Remaining columns: {list(df.columns)}")
    else:
        print("All_Employees column not found; skipped.")

    # TASK 27: Delete 'Personal_Consumption_Expenditures' and 'National_Home_Price_Index'
    show_section("Task 27: Drop Personal_Consumption_Expenditures & National_Home_Price_Index (if present)")
    drop_cols = [c for c in [find_col(df,"Personal_Consumption_Expenditures"), find_col(df,"National_Home_Price_Index")] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print("Dropped:", drop_cols)
    else:
        print("One or both columns not present; skipped.")

    # TASK 28: Rename Personal_Consumption_Expenditures -> changed name
    show_section("Task 28: Rename column Personal_Consumption_Expenditures -> Personal_Consumption_Expenditures_Changed")
    pce = find_col(df, "Personal_Consumption_Expenditures")
    if pce and pce in df.columns:
        df = df.rename(columns={pce: "Personal_Consumption_Expenditures_Changed"})
        print("Renamed. Columns now:", list(df.columns))
    else:
        print("Original column not found to rename (maybe already dropped).")

    # TASK 29: Rename index label from 5 to 8
    show_section("Task 29: Rename index label 5 -> 8 (if 5 exists)")
    if 5 in df.index:
        if 8 in df.index:
            print("Note: label 8 already exists — renaming may create duplicates.")
        df = df.rename(index={5:8})
        print("Index labels sample (first 20):", list(df.index)[:20])
    else:
        print("Label 5 not present; skipped index rename.")

    # TASK 30: query: Industrial_Production <=0.5 & Consumer_Price_Index >0.2 & Year==1992
    show_section("Task 30: query Industrial_Production <=0.5 & Consumer_Price_Index >0.2 & Year == 1992")
    ip = find_col(df, "Industrial_Production")
    cpi = find_col(df, "Consumer_Price Index")
    yearcol = find_col(df, "Year")
    if ip and cpi and yearcol:
        cond = (df[ip] <= 0.5) & (df[cpi] > 0.2) & (df[yearcol] == 1992)
        res = df.loc[cond]
        print(res.to_string() if not res.empty else "No rows match the query.")
    else:
        print("Required columns for query not found.")

    # TASK 31: sort by Consumer_Price_Index ascending
    show_section("Task 31: Sort DataFrame by Consumer_Price_Index ascending")
    if cpi:
        sorted_df = df.sort_values(by=cpi, ascending=True)
        print(sorted_df[[cpi]].head(10).to_string())
    else:
        print("Consumer_Price_Index column not found; cannot sort.")

    # TASK 32: group by Year and sum National_Home_Price_Index
    show_section("Task 32: Group by Year and sum National_Home_Price_Index")
    nhpi = find_col(df, "National_Home_Price_Index")
    if yearcol and nhpi and nhpi in df.columns:
        grp = df.groupby(yearcol)[nhpi].sum().reset_index()
        print(grp.to_string(index=False))
    else:
        print("Year or National_Home_Price_Index column missing; cannot group.")

    # TASK 33: dropna() remove rows with any missing values
    show_section("Task 33: dropna() - remove rows with any NaN")
    df_dropna = df.dropna(how="any").reset_index(drop=True)
    print("After dropna rows:", len(df_dropna))
    print(df_dropna.head(5).to_string())
    short_analysis("dropna(how='any') removes rows with any missing value; use with care if many NaNs exist.")

    # TASK 34: filling NaN with 0
    show_section("Task 34: fillna(0) - fill missing values with 0")
    df_filled = df.fillna(0)
    print("After fillna(0) sample (first 5 rows):\n", df_filled.head(5).to_string())

    show_section("All tasks completed")
    print("Notes: If any column is not found, check printed mapping above and open CSV to see exact header names.")
    print("If index labels used in .loc are missing, you can reset index: df = df.reset_index(drop=True)")

if __name__ == "__main__":
    main()

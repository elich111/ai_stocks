import csv, io, requests, sys, time

print("Hello from Codex")

# Source file for all non-Nasdaq tickers
URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
out_path = "nyse_all_with_desc.txt"

# Step 1: get otherlisted.txt
r = requests.get(URL, timeout=30)
r.raise_for_status()
data = [line for line in r.text.splitlines() if not line.startswith("File Creation Time")]
reader = csv.DictReader(io.StringIO("\n".join(data)), delimiter='|')

# Filter for NYSE only (Exchange = "N")
nyse = [(row["ACT Symbol"], row["Security Name"]) for row in reader 
        if row.get("Exchange") == "N" and row.get("Test Issue") == "N"]

def get_description(ticker):
    """
    Fetch short business summary from Yahoo Finance API (unofficial).
    """
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=assetProfile"
    try:
        resp = requests.get(url, timeout=10).json()
        profile = resp["quoteSummary"]["result"][0]["assetProfile"]
        return profile.get("longBusinessSummary","").replace("\n"," ").strip()
    except Exception:
        return ""

rows = []
for i, (ticker, name) in enumerate(nyse, 1):
    desc = get_description(ticker)
    rows.append(f"{ticker};{name};{desc}")
    # polite delay to avoid throttling
    time.sleep(0.5)
    print(f"[{i}/{len(nyse)}] {ticker}")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(rows))

print(f"Done. Wrote {len(rows)} lines to {out_path}")

import yfinance as yf

tech_keywords = [
    'technology', 'software', 'internet', 'cloud', 'data', 'analytics',
    'semiconductor', 'computer', 'hardware', 'electronics', 'biotech',
    'healthtech', 'digital', 'online', 'network', 'security', 'systems',
    'communications', 'information', 'platform', 'online', 'e-commerce',
    'artificial intelligence', 'ai', 'automation', 'robotics', 'saas',
    'infrastructure', 'development', 'quantum', 'solutions'
]

import yfinance as yf
from tqdm import tqdm

tech_keywords = [
    'technology', 'software', 'internet', 'cloud', 'data', 'analytics',
    'semiconductor', 'computer', 'hardware', 'electronics', 'biotech',
    'healthtech', 'digital', 'online', 'network', 'security', 'systems',
    'communications', 'information', 'platform', 'online', 'e-commerce',
    'artificial intelligence', 'ai', 'automation', 'robotics', 'saas',
    'infrastructure', 'development', 'quantum', 'solutions'
]

file_path = "d:\\projects\\ai_stock\\nyse_all_with_desc.txt"
file_path_tech = "d:\\projects\\ai_stock\\nyse_tech_companies_with_desc.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

tech_companies = []
for line in tqdm(lines, desc="Filtering tickers"):
    parts = line.strip().split(';')
    if len(parts) > 0:
        ticker = parts[0]
        try:
            ticker_info = yf.Ticker(ticker).info
            description = ticker_info.get('longBusinessSummary', '').lower()
            if any(keyword in description for keyword in tech_keywords):
                tech_companies.append(f"{ticker};{ticker_info.get('longBusinessSummary', '')};tech\n")
        except Exception as e:
            print(f"Could not process {ticker}: {e}")


with open(file_path_tech, 'w', encoding='utf-8') as f:
    f.writelines(tech_companies)

print(f'Filtered {len(lines)} companies down to {len(tech_companies)} tech companies.')

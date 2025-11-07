# How to Get Real IBM 10-K Reports from SEC EDGAR

Since SEC EDGAR has strict rate limiting and header requirements, here's the easiest manual approach:

## Option 1: Direct Download URLs (Fastest!)

IBM's most recent 10-K filings are available at these direct links:

### 2022 Report (for fiscal year 2022, filed Feb 2023)
```
https://www.sec.gov/ix?doc=/Archives/edgar/data/0000051143/000155837023002376/ibm-20221231x10k.htm
```

### 2021 Report  
```
https://www.sec.gov/ix?doc=/Archives/edgar/data/0000051143/000155837022002426/ibm-20211231x10k.htm
```

### 2020 Report
```
https://www.sec.gov/ix?doc=/Archives/edgar/data/0000051143/000155837021001426/ibm-20201231x10k.htm
```

### 2019 Report
```
https://www.sec.gov/ix?doc=/Archives/edgar/data/0000051143/000155837020001267/ibm-20191231x10k.htm
```

## Option 2: Manual Download Steps

1. **Visit SEC EDGAR IBM page:**
   ```
   https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000051143&type=10-K&dateb=&owner=exclude&count=100
   ```

2. **Find the 10-K filing** for each year you want (2016-2024)

3. **Click "Documents"** button for each filing

4. **Right-click the HTML file** (usually ends in `10k.htm`) and "Save As..."

5. **Save to a directory** like `~/ibm_reports/`

## Option 3: Use Our Python Script

Once you have the HTML files downloaded manually:

```bash
# Save them to a directory
mkdir -p ~/ibm_reports_html

# Then use our ingest script
python ingest_real_reports.py --directory ~/ibm_reports_html --clear-first
```

## Option 4: Browser Method (Quick Test)

1. Open: https://www.sec.gov/ix?doc=/Archives/edgar/data/0000051143/000155837023002376/ibm-20221231x10k.htm

2. Press `Cmd+S` (Mac) or `Ctrl+S` (Windows) to save

3. Save as `IBM_2022_10K.html`

4. Repeat for other years

5. Run our ingestion script

## Quick Test with 2022 Report

```bash
cd /Users/robertbrodie/embeddingspin

# Download just the 2022 report
curl -o IBM_2022_10K.htm \
  -H "User-Agent: research@example.com" \
  "https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/ibm-20221231x10k.htm"

# Create directory and move file
mkdir -p ~/ibm_test
mv IBM_2022_10K.htm ~/ibm_test/

# Ingest it
source venv/bin/activate
python ingest_real_reports.py --directory ~/ibm_test --clear-first
```

## What's in a 10-K Report?

Each 10-K contains:
- Business description
- Risk factors  
- Financial statements (balance sheet, income statement, cash flow)
- Management discussion & analysis (MD&A)
- Revenue breakdowns by segment
- Strategic initiatives
- Forward-looking statements

Perfect for temporal spin retrieval!

## After Ingestion

Once ingested, query them:

```bash
python demo.py --query "2022 revenue" --timestamp 2022-12-31 --beta 10
python demo.py --query "Red Hat acquisition impact" --timestamp 2019-12-31 --beta 5
python demo.py --query "cloud revenue growth" --timestamp 2021-06-30 --beta 8
```

---

**Note:** SEC EDGAR's JSON API requires specific headers and rate limiting. For research purposes, manual download of 8-10 files is actually faster and more reliable than automated scraping.


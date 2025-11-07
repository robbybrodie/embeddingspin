#!/bin/bash
echo "Downloading IBM 10-K reports with browser headers..."
echo ""

# Function to download with browser-like headers
download_report() {
  year=$1
  url=$2
  curl -L -s \
    -H "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36" \
    -H "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8" \
    -H "Accept-Language: en-US,en;q=0.9" \
    -H "Accept-Encoding: identity" \
    -H "Connection: keep-alive" \
    -o "IBM_${year}_10K.htm" \
    "$url"
  
  size=$(ls -lh "IBM_${year}_10K.htm" | awk '{print $5}')
  echo "✓ $year ($size)"
  sleep 1
}

download_report 2022 "https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/ibm-20221231x10k.htm"
download_report 2021 "https://www.sec.gov/Archives/edgar/data/51143/000155837022002426/ibm-20211231x10k.htm"
download_report 2020 "https://www.sec.gov/Archives/edgar/data/51143/000155837021001426/ibm-20201231x10k.htm"
download_report 2019 "https://www.sec.gov/Archives/edgar/data/51143/000155837020001267/ibm-20191231x10k.htm"
download_report 2018 "https://www.sec.gov/Archives/edgar/data/51143/000155837019001112/ibm-20181231x10k.htm"
download_report 2017 "https://www.sec.gov/Archives/edgar/data/51143/000155837018001146/ibm-20171231x10k.htm"
download_report 2016 "https://www.sec.gov/Archives/edgar/data/51143/000104746917000575/a2230647z10-k.htm"
download_report 2015 "https://www.sec.gov/Archives/edgar/data/51143/000104746916010329/a2227651z10-k.htm"
download_report 2014 "https://www.sec.gov/Archives/edgar/data/51143/000104746915001116/a2223279z10-k.htm"

echo ""
echo "✓ Downloads complete!"
echo ""
echo "Total files:"
ls -lh IBM_*.htm


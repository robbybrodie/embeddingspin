#!/bin/bash
echo "Downloading IBM 10-K reports using wget with browser headers..."
echo ""

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  --header="Accept: text/html" \
  --header="Accept-Language: en-US" \
  -q --show-progress \
  -O IBM_2022_10K.htm \
  "https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/ibm-20221231x10k.htm" && echo "✓ 2022"

sleep 1

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  --header="Accept: text/html" \
  -q --show-progress \
  -O IBM_2021_10K.htm \
  "https://www.sec.gov/Archives/edgar/data/51143/000155837022002426/ibm-20211231x10k.htm" && echo "✓ 2021"

sleep 1

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  --header="Accept: text/html" \
  -q --show-progress \
  -O IBM_2020_10K.htm \
  "https://www.sec.gov/Archives/edgar/data/51143/000155837021001426/ibm-20201231x10k.htm" && echo "✓ 2020"

sleep 1

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  --header="Accept: text/html" \
  -q --show-progress \
  -O IBM_2019_10K.htm \
  "https://www.sec.gov/Archives/edgar/data/51143/000155837020001267/ibm-20191231x10k.htm" && echo "✓ 2019"

sleep 1

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  --header="Accept: text/html" \
  -q --show-progress \
  -O IBM_2018_10K.htm \
  "https://www.sec.gov/Archives/edgar/data/51143/000155837019001112/ibm-20181231x10k.htm" && echo "✓ 2018"

sleep 1

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  --header="Accept: text/html" \
  -q --show-progress \
  -O IBM_2017_10K.htm \
  "https://www.sec.gov/Archives/edgar/data/51143/000155837018001146/ibm-20171231x10k.htm" && echo "✓ 2017"

sleep 1

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  --header="Accept: text/html" \
  -q --show-progress \
  -O IBM_2016_10K.htm \
  "https://www.sec.gov/Archives/edgar/data/51143/000104746917000575/a2230647z10-k.htm" && echo "✓ 2016"

sleep 1

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  --header="Accept: text/html" \
  -q --show-progress \
  -O IBM_2015_10K.htm \
  "https://www.sec.gov/Archives/edgar/data/51143/000104746916010329/a2227651z10-k.htm" && echo "✓ 2015"

sleep 1

wget --user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" \
  --header="Accept: text/html" \
  -q --show-progress \
  -O IBM_2014_10K.htm \
  "https://www.sec.gov/Archives/edgar/data/51143/000104746915001116/a2223279z10-k.htm" && echo "✓ 2014"

echo ""
echo "Download complete!"
ls -lh IBM_*.htm


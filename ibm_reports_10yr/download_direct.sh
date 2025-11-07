#!/bin/bash
echo "Downloading IBM 10-K HTML documents directly..."
echo "This may take 1-2 minutes..."
echo ""

# Use the actual 10-K document paths (these are the full HTML files)
curl -s "https://www.sec.gov/Archives/edgar/data/51143/000155837023002376/ibm-20221231x10k.htm" -o IBM_2022_10K.htm && echo "✓ 2022 ($(ls -lh IBM_2022_10K.htm | awk '{print $5}'))"
sleep 1

curl -s "https://www.sec.gov/Archives/edgar/data/51143/000155837022002426/ibm-20211231x10k.htm" -o IBM_2021_10K.htm && echo "✓ 2021 ($(ls -lh IBM_2021_10K.htm | awk '{print $5}'))"
sleep 1

curl -s "https://www.sec.gov/Archives/edgar/data/51143/000155837021001426/ibm-20201231x10k.htm" -o IBM_2020_10K.htm && echo "✓ 2020 ($(ls -lh IBM_2020_10K.htm | awk '{print $5}'))"
sleep 1

curl -s "https://www.sec.gov/Archives/edgar/data/51143/000155837020001267/ibm-20191231x10k.htm" -o IBM_2019_10K.htm && echo "✓ 2019 ($(ls -lh IBM_2019_10K.htm | awk '{print $5}'))"
sleep 1

curl -s "https://www.sec.gov/Archives/edgar/data/51143/000155837019001112/ibm-20181231x10k.htm" -o IBM_2018_10K.htm && echo "✓ 2018 ($(ls -lh IBM_2018_10K.htm | awk '{print $5}'))"
sleep 1

curl -s "https://www.sec.gov/Archives/edgar/data/51143/000155837018001146/ibm-20171231x10k.htm" -o IBM_2017_10K.htm && echo "✓ 2017 ($(ls -lh IBM_2017_10K.htm | awk '{print $5}'))"
sleep 1

curl -s "https://www.sec.gov/Archives/edgar/data/51143/000104746917000575/a2230647z10-k.htm" -o IBM_2016_10K.htm && echo "✓ 2016 ($(ls -lh IBM_2016_10K.htm | awk '{print $5}'))"
sleep 1

curl -s "https://www.sec.gov/Archives/edgar/data/51143/000104746916010329/a2227651z10-k.htm" -o IBM_2015_10K.htm && echo "✓ 2015 ($(ls -lh IBM_2015_10K.htm | awk '{print $5}'))"
sleep 1

curl -s "https://www.sec.gov/Archives/edgar/data/51143/000104746915001116/a2223279z10-k.htm" -o IBM_2014_10K.htm && echo "✓ 2014 ($(ls -lh IBM_2014_10K.htm | awk '{print $5}'))"

echo ""
echo "✓ All downloads complete!"
echo ""
echo "Files:"
ls -lh IBM_*.htm


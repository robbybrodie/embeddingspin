#!/bin/bash
echo "Downloading IBM 10-K reports with correct URLs..."
echo ""

# 2023 - Need to find actual filing
echo "2023: Searching for correct URL..."
curl -s "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000051143&type=10-K&dateb=&owner=exclude&count=10&output=atom" | grep -o "Archives/edgar/data/51143/[^\"]*10k.htm" | head -1 | while read path; do
  curl -s -o IBM_2023_10K.htm "https://www.sec.gov/$path" && echo "✓ Downloaded 2023" && ls -lh IBM_2023_10K.htm
done

# For the others, let's use a known working pattern
echo ""
echo "Downloading 2014-2022 using iXBRL viewer URLs..."

# These are the inline XBRL viewer URLs which work better
curl -L -s -o IBM_2022_10K.htm "https://www.sec.gov/cgi-bin/viewer?action=view&cik=51143&accession_number=0001558370-23-002376&xbrl_type=v" && echo "✓ 2022" && ls -lh IBM_2022_10K.htm

sleep 1

curl -L -s -o IBM_2021_10K.htm "https://www.sec.gov/cgi-bin/viewer?action=view&cik=51143&accession_number=0001558370-22-002426&xbrl_type=v" && echo "✓ 2021" && ls -lh IBM_2021_10K.htm

sleep 1

curl -L -s -o IBM_2020_10K.htm "https://www.sec.gov/cgi-bin/viewer?action=view&cik=51143&accession_number=0001558370-21-001426&xbrl_type=v" && echo "✓ 2020" && ls -lh IBM_2020_10K.htm

sleep 1

curl -L -s -o IBM_2019_10K.htm "https://www.sec.gov/cgi-bin/viewer?action=view&cik=51143&accession_number=0001558370-20-001267&xbrl_type=v" && echo "✓ 2019" && ls -lh IBM_2019_10K.htm

sleep 1

curl -L -s -o IBM_2018_10K.htm "https://www.sec.gov/cgi-bin/viewer?action=view&cik=51143&accession_number=0001558370-19-001112&xbrl_type=v" && echo "✓ 2018" && ls -lh IBM_2018_10K.htm

sleep 1

curl -L -s -o IBM_2017_10K.htm "https://www.sec.gov/cgi-bin/viewer?action=view&cik=51143&accession_number=0001558370-18-001146&xbrl_type=v" && echo "✓ 2017" && ls -lh IBM_2017_10K.htm

sleep 1

# 2014-2016 use older format
curl -s -o IBM_2016_10K.htm "https://www.sec.gov/Archives/edgar/data/51143/000104746917000575/0001047469-17-000575-index.htm" && echo "✓ 2016" && ls -lh IBM_2016_10K.htm

sleep 1

curl -s -o IBM_2015_10K.htm "https://www.sec.gov/Archives/edgar/data/51143/000104746916010329/0001047469-16-010329-index.htm" && echo "✓ 2015" && ls -lh IBM_2015_10K.htm

sleep 1

curl -s -o IBM_2014_10K.htm "https://www.sec.gov/Archives/edgar/data/51143/000104746915001116/0001047469-15-001116-index.htm" && echo "✓ 2014" && ls -lh IBM_2014_10K.htm

echo ""
echo "✓ Download complete!"
echo ""
ls -lh IBM_*.htm | awk '{print $9, $5}'


#!/usr/bin/env python3
"""
SEC EDGAR IBM 10-K Downloader and Ingestion
============================================

Automatically downloads IBM 10-K annual reports from SEC EDGAR and ingests
them into the temporal spin retrieval system.

This script:
1. Queries SEC EDGAR API for IBM 10-K filings (2016-2024)
2. Downloads and parses the HTML/iXBRL content
3. Extracts clean text from the filings
4. Ingests with temporal spin encoding

Usage:
    # Download and ingest with mock embeddings (fast testing)
    python download_sec_reports.py
    
    # With real LlamaStack embeddings
    python download_sec_reports.py --use-llamastack
    
    # Specific years only
    python download_sec_reports.py --years 2019 2020 2021 2022
    
    # Save to persistent Chroma database
    python download_sec_reports.py --vector-store chroma
"""

import argparse
import os
import sys
import re
import time
from datetime import datetime, timezone
from typing import List, Tuple, Optional
import json

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("ERROR: Required packages not installed")
    print("Install with: pip install requests beautifulsoup4")
    sys.exit(1)

from llamastack_client import LlamaStackEmbeddingClient, MockEmbeddingClient
from vector_store import InMemoryVectorStore, ChromaVectorStore
from ingestion import TemporalSpinIngestionPipeline


# SEC EDGAR configuration
SEC_BASE_URL = "https://www.sec.gov"
IBM_CIK = "0000051143"
USER_AGENT = "Temporal-Spin-Retrieval research@example.com"  # SEC requires user agent

# Headers for SEC requests (required by SEC)
SEC_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov"
}


def get_ibm_10k_filings(start_year: int = 2016, end_year: int = 2024) -> List[Tuple[int, str]]:
    """
    Get IBM 10-K filing URLs from SEC EDGAR.
    
    Returns:
        List of (year, document_url) tuples
    """
    print(f"Fetching IBM 10-K filings from SEC EDGAR ({start_year}-{end_year})...")
    
    # Use SEC EDGAR API to get submissions
    # API needs CIK with 10 digits (padded with leading zeros)
    cik_padded = IBM_CIK.zfill(10)
    api_url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    
    try:
        response = requests.get(api_url, headers=SEC_HEADERS)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"ERROR: Failed to fetch from SEC API: {e}")
        return []
    
    # Parse recent filings
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accession_numbers = filings.get("accessionNumber", [])
    filing_dates = filings.get("filingDate", [])
    primary_docs = filings.get("primaryDocument", [])
    
    results = []
    
    for i, form in enumerate(forms):
        if form != "10-K":
            continue
        
        # Get filing date and extract year
        filing_date = filing_dates[i]
        year = int(filing_date.split("-")[0])
        
        # Check if in our year range
        # Note: 10-K for fiscal year 2022 is filed in early 2023
        fiscal_year = year - 1  # 10-K filed in 2023 is for fiscal 2022
        
        if fiscal_year < start_year or fiscal_year > end_year:
            continue
        
        # Build document URL
        accession = accession_numbers[i].replace("-", "")
        primary_doc = primary_docs[i]
        doc_url = f"{SEC_BASE_URL}/Archives/edgar/data/{IBM_CIK.lstrip('0')}/{accession}/{primary_doc}"
        
        results.append((fiscal_year, doc_url))
        print(f"  Found: {fiscal_year} 10-K - {doc_url}")
    
    # Sort by year
    results.sort(key=lambda x: x[0])
    
    return results


def download_and_parse_10k(url: str) -> Optional[str]:
    """
    Download and parse 10-K HTML document.
    
    Extracts the main text content, removing scripts, styles, and navigation.
    
    Args:
        url: SEC EDGAR document URL
    
    Returns:
        Cleaned text content
    """
    print(f"  Downloading: {url}")
    
    try:
        # Be respectful to SEC servers
        time.sleep(0.5)
        
        response = requests.get(url, headers=SEC_HEADERS, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit to reasonable size (first 100k chars contains most important info)
        if len(text) > 100000:
            text = text[:100000]
        
        print(f"  ✓ Extracted {len(text)} characters")
        return text
        
    except Exception as e:
        print(f"  ERROR: Failed to download/parse: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Download IBM 10-K reports from SEC EDGAR and ingest into temporal spin system"
    )
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        default=None,
        help="Specific years to download (default: 2016-2024)"
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=2016,
        help="Start year (default: 2016)"
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=2024,
        help="End year (default: 2024)"
    )
    parser.add_argument(
        '--use-llamastack',
        action='store_true',
        help="Use LlamaStack for embeddings (default: mock embeddings)"
    )
    parser.add_argument(
        '--llamastack-url',
        type=str,
        default=os.getenv('LLAMASTACK_URL', 'http://localhost:8000'),
        help="LlamaStack API URL"
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default=os.getenv('EMBEDDING_MODEL', 'text-embedding-v1'),
        help="Embedding model name"
    )
    parser.add_argument(
        '--vector-store',
        type=str,
        choices=['memory', 'chroma'],
        default='memory',
        help="Vector store type (default: memory)"
    )
    parser.add_argument(
        '--chroma-dir',
        type=str,
        default='./ibm_sec_chroma',
        help="Chroma persistence directory"
    )
    parser.add_argument(
        '--save-text',
        action='store_true',
        help="Save downloaded text to files for inspection"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("IBM 10-K DOWNLOADER - SEC EDGAR")
    print("="*80)
    print()
    print("⚠️  IMPORTANT: SEC EDGAR Terms of Use")
    print("This script complies with SEC fair access rules:")
    print("  • Identifies itself with User-Agent")
    print("  • Rate limits requests (0.5s between downloads)")
    print("  • For research/educational purposes")
    print()
    
    # Get filings
    if args.years:
        print(f"Target years: {args.years}")
        start_year = min(args.years)
        end_year = max(args.years)
    else:
        start_year = args.start_year
        end_year = args.end_year
        print(f"Target years: {start_year}-{end_year}")
    
    print()
    filings = get_ibm_10k_filings(start_year, end_year)
    
    if not filings:
        print("\nERROR: No filings found!")
        sys.exit(1)
    
    # Filter to specific years if requested
    if args.years:
        filings = [(year, url) for year, url in filings if year in args.years]
    
    print(f"\n✓ Found {len(filings)} filings to download")
    print()
    
    # Download and parse
    print("Downloading and parsing 10-K documents...")
    print("-"*80)
    
    reports = []
    for year, url in filings:
        print(f"\n[{year}]")
        text = download_and_parse_10k(url)
        
        if text:
            timestamp = datetime(year, 12, 31, tzinfo=timezone.utc)
            doc_id = f"ibm-sec-10k-{year}"
            reports.append((text, timestamp, doc_id, url))
            
            # Save to file if requested
            if args.save_text:
                filename = f"IBM_10K_{year}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"  Saved to: {filename}")
    
    if not reports:
        print("\nERROR: No reports successfully downloaded!")
        sys.exit(1)
    
    print()
    print(f"✓ Successfully downloaded {len(reports)} reports")
    print()
    
    # Initialize embedding client
    print("-"*80)
    if args.use_llamastack:
        print(f"Using LlamaStack: {args.llamastack_url}")
        print(f"Model: {args.embedding_model}")
        embedding_client = LlamaStackEmbeddingClient(
            base_url=args.llamastack_url,
            model_name=args.embedding_model
        )
    else:
        print("Using Mock Embeddings (for testing)")
        embedding_client = MockEmbeddingClient(dimension=384)
    
    # Initialize vector store
    if args.vector_store == 'chroma':
        print(f"Vector Store: Chroma (persistent at {args.chroma_dir})")
        vector_store = ChromaVectorStore(
            collection_name="ibm_sec_10k",
            persist_directory=args.chroma_dir
        )
    else:
        print("Vector Store: In-Memory")
        vector_store = InMemoryVectorStore()
    
    # Clear existing data
    print("\nClearing existing data...")
    vector_store.clear()
    print("✓ Data cleared")
    print()
    
    # Create ingestion pipeline
    pipeline = TemporalSpinIngestionPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store
    )
    
    # Ingest reports
    print("Ingesting reports with temporal spin encoding...")
    print("(This may take a while with real embeddings)")
    print("-"*80)
    
    texts = [text for text, _, _, _ in reports]
    timestamps = [ts for _, ts, _, _ in reports]
    doc_ids = [doc_id for _, _, doc_id, _ in reports]
    metadatas = [{"source": "SEC EDGAR", "url": url} for _, _, _, url in reports]
    
    try:
        docs = pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps,
            doc_ids=doc_ids,
            metadatas=metadatas
        )
        
        print()
        print("="*80)
        print("INGESTION COMPLETE ✓")
        print("="*80)
        print(f"✓ Ingested {len(docs)} IBM 10-K reports from SEC EDGAR")
        print()
        
        # Show summary
        print("Documents in vector store:")
        for doc in sorted(docs, key=lambda d: d.timestamp):
            year = doc.timestamp.year
            phi_deg = doc.phi * 180 / 3.14159
            text_len = len(doc.text)
            print(f"  • {year}: Phase φ = {phi_deg:6.1f}°, {text_len:,} chars, ID = {doc.doc_id}")
        
        print()
        print("="*80)
        print("READY TO QUERY!")
        print("="*80)
        print()
        print("Example queries:")
        print("  python demo.py --query 'IBM revenue 2019' --timestamp 2019-12-31 --beta 10")
        print("  python demo.py --query 'Red Hat acquisition' --timestamp 2019-06-30 --beta 5")
        print("  python demo.py --query 'cloud strategy' --timestamp 2022-01-01 --beta 8")
        print()
        print("Or start the API server:")
        print("  python api.py")
        print()
        
    except Exception as e:
        print(f"\nERROR during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


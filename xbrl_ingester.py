#!/usr/bin/env python3
"""
XBRL ZIP File Ingester for Temporal-Phase Spin Retrieval
=========================================================

Extracts and ingests SEC XBRL filings (10-K reports) into the temporal spin
retrieval system.

Key Features:
- Extracts iXBRL/HTML documents from XBRL ZIP packages
- Prioritizes document content over filename metadata for fiscal year detection
- Properly handles XBRL metadata (dei:DocumentPeriodEndDate, dei:DocumentFiscalYearFocus)
- Cleans and extracts narrative text from structured filings

Usage:
    python xbrl_ingester.py --directory ./sample10ks --clear-first
"""

import argparse
import os
import sys
import zipfile
import re
from pathlib import Path
from datetime import datetime, timezone
from bs4 import BeautifulSoup

from llamastack_client import LlamaStackEmbeddingClient, MockEmbeddingClient
from vector_store import InMemoryVectorStore, ChromaVectorStore
from ingestion import TemporalSpinIngestionPipeline


def extract_fiscal_year_from_xbrl(zip_path):
    """
    Extract fiscal year from XBRL ZIP file.
    
    Priority order:
    1. XBRL metadata (dei:DocumentPeriodEndDate, dei:DocumentFiscalYearFocus)
    2. Document title/cover page content
    3. Filename patterns
    4. Accession number (fallback with warning)
    
    Returns:
        (year, source) tuple - year as int, source as string for debugging
    """
    zip_path = Path(zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        filenames = zf.namelist()
        
        # Find the main HTML/iXBRL document
        # Usually named like: ibm-20231231.htm, ibm-20231231x10k.htm, or similar
        html_file = None
        for fname in filenames:
            if fname.endswith('.htm') or fname.endswith('.html'):
                # Prioritize files with '10k' in the name
                if '10k' in fname.lower():
                    html_file = fname
                    break
                # Otherwise take the first .htm file
                if not html_file:
                    html_file = fname
        
        if not html_file:
            raise ValueError(f"No HTML/iXBRL document found in {zip_path.name}")
        
        # Read the HTML content
        with zf.open(html_file) as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Method 1: Look for XBRL metadata tags (HIGHEST PRIORITY)
        # dei:DocumentPeriodEndDate (e.g., 2023-12-31)
        period_end = soup.find(['dei:documentperiodenddate', 'dei:DocumentPeriodEndDate'])
        if period_end:
            date_text = period_end.get_text().strip()
            match = re.search(r'(\d{4})-\d{2}-\d{2}', date_text)
            if match:
                year = int(match.group(1))
                print(f"  Found fiscal year from XBRL DocumentPeriodEndDate: {year}")
                return year, "XBRL_DocumentPeriodEndDate"
        
        # dei:DocumentFiscalYearFocus (e.g., 2023)
        fiscal_year = soup.find(['dei:documentfiscalyearfocus', 'dei:DocumentFiscalYearFocus'])
        if fiscal_year:
            year_text = fiscal_year.get_text().strip()
            if year_text.isdigit():
                year = int(year_text)
                print(f"  Found fiscal year from XBRL DocumentFiscalYearFocus: {year}")
                return year, "XBRL_DocumentFiscalYearFocus"
        
        # Method 2: Parse document title
        # Look for patterns like "Form 10-K For the fiscal year ended December 31, 2023"
        text_content = soup.get_text()
        
        # Pattern: "fiscal year ended [Month] [Day], [Year]"
        match = re.search(r'fiscal\s+year\s+ended[^\d]*(\d{4})', text_content, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            print(f"  Found fiscal year from document text: {year}")
            return year, "Document_Text"
        
        # Pattern: "For the year ended December 31, 2023"
        match = re.search(r'year\s+ended[^\d]*\d{1,2},?\s*(\d{4})', text_content, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            print(f"  Found fiscal year from document text: {year}")
            return year, "Document_Text"
        
        # Method 3: Extract from main HTML filename
        # Pattern: ibm-20231231x10k.htm → 2023
        match = re.search(r'(\d{4})\d{4}', html_file)
        if match:
            year = int(match.group(1))
            print(f"  Found fiscal year from HTML filename: {year}")
            return year, "HTML_Filename"
    
    # Method 4: Extract from ZIP filename (FALLBACK)
    # Pattern: accession number like 0001558370-23-002376-xbrl.zip
    # The '23' means filed in 2023, so likely fiscal year 2022
    match = re.search(r'-(\d{2})-\d+', zip_path.name)
    if match:
        file_year = 2000 + int(match.group(1))
        # Assume fiscal year is one year before filing year
        year = file_year - 1
        print(f"  WARNING: Using accession number fallback (filed {file_year}, assuming fiscal {year})")
        return year, f"Accession_Fallback_(filed_{file_year})"
    
    raise ValueError(f"Could not determine fiscal year from {zip_path.name}")


def extract_text_from_xbrl_zip(zip_path):
    """
    Extract clean text from XBRL ZIP file.
    
    Returns:
        (text, fiscal_year, year_source) tuple
    """
    zip_path = Path(zip_path)
    
    print(f"Processing: {zip_path.name}")
    
    # Extract fiscal year first
    fiscal_year, year_source = extract_fiscal_year_from_xbrl(zip_path)
    
    # Now extract the text content
    with zipfile.ZipFile(zip_path, 'r') as zf:
        filenames = zf.namelist()
        
        # Find main HTML document
        html_file = None
        for fname in filenames:
            if fname.endswith('.htm') or fname.endswith('.html'):
                if '10k' in fname.lower():
                    html_file = fname
                    break
                if not html_file:
                    html_file = fname
        
        if not html_file:
            raise ValueError(f"No HTML document found in {zip_path.name}")
        
        # Extract and parse HTML
        with zf.open(html_file) as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style, and hidden XBRL metadata tags
        for element in soup(['script', 'style', 'ix:hidden', 'ix:continuation']):
            element.decompose()
        
        # Get clean text
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        
        print(f"  ✓ Extracted {len(text)} chars, Fiscal Year: {fiscal_year} (source: {year_source})")
        
        return text, fiscal_year, year_source


def load_xbrl_reports_from_directory(directory):
    """
    Load all XBRL ZIP files from a directory.
    
    Returns:
        List of (text, timestamp, doc_id, filepath, year_source) tuples
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        return []
    
    reports = []
    
    # Look for XBRL ZIP files
    for filepath in sorted(directory.glob("*.zip")):
        if 'xbrl' not in filepath.name.lower():
            print(f"Skipping non-XBRL file: {filepath.name}")
            continue
        
        try:
            text, fiscal_year, year_source = extract_text_from_xbrl_zip(filepath)
            
            # Create timestamp (December 31 of the fiscal year)
            timestamp = datetime(fiscal_year, 12, 31, tzinfo=timezone.utc)
            
            # Create doc ID
            doc_id = f"ibm-10k-{fiscal_year}"
            
            reports.append((text, timestamp, doc_id, str(filepath), year_source))
            
        except Exception as e:
            print(f"  ERROR processing {filepath.name}: {e}")
            continue
    
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Ingest XBRL ZIP files into temporal spin system"
    )
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help="Directory containing XBRL ZIP files"
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
        default='./chroma_db',
        help="Chroma persistence directory"
    )
    parser.add_argument(
        '--clear-first',
        action='store_true',
        help="Clear existing data before ingesting"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("XBRL ZIP FILE INGESTION")
    print("="*80)
    print()
    
    # Initialize embedding client
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
            collection_name="ibm_annual_reports",
            persist_directory=args.chroma_dir
        )
    else:
        print("Vector Store: In-Memory")
        vector_store = InMemoryVectorStore()
    
    # Clear existing data if requested
    if args.clear_first:
        print("\nClearing existing vector store data...")
        vector_store.clear()
        print("✓ Data cleared")
    
    # Load XBRL reports
    print(f"\nLoading XBRL reports from: {args.directory}")
    print("-"*80)
    reports = load_xbrl_reports_from_directory(args.directory)
    
    if not reports:
        print("\nERROR: No XBRL reports found!")
        print("\nMake sure:")
        print("  1. The directory exists")
        print("  2. It contains *-xbrl.zip files")
        print("  3. Files are valid XBRL packages")
        sys.exit(1)
    
    print(f"\n✓ Found {len(reports)} reports")
    print()
    
    # Create ingestion pipeline
    pipeline = TemporalSpinIngestionPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store
    )
    
    # Ingest reports
    print("Ingesting reports with temporal-phase spin encoding...")
    print("-"*80)
    
    texts = [text for text, _, _, _, _ in reports]
    timestamps = [ts for _, ts, _, _, _ in reports]
    doc_ids = [doc_id for _, _, doc_id, _, _ in reports]
    metadatas = [
        {
            "filepath": fp,
            "year_source": ys,
            "fiscal_year": ts.year
        }
        for _, ts, _, fp, ys in reports
    ]
    
    try:
        docs = pipeline.ingest_batch(
            texts=texts,
            timestamps=timestamps,
            doc_ids=doc_ids,
            metadatas=metadatas
        )
        
        print()
        print("="*80)
        print("INGESTION COMPLETE")
        print("="*80)
        print(f"✓ Ingested {len(docs)} documents")
        print()
        
        # Show summary with temporal spin details
        print("Documents in vector store (with temporal-phase spin encoding):")
        for doc in docs:
            year = doc.timestamp.year
            phi_deg = doc.phi * 180 / 3.14159
            year_source = doc.metadata.get('year_source', 'unknown')
            print(f"  • {year}: φ = {phi_deg:.1f}°, spin = [{doc.spin_vector[0]:.3f}, {doc.spin_vector[1]:.3f}]")
            print(f"           Year source: {year_source}, ID: {doc.doc_id}")
        
        print()
        print("✓ Temporal-phase spin encoding complete!")
        print("  Each document's timestamp is encoded as an angular position on the unit circle.")
        print("  The β (beta) parameter controls temporal zoom during retrieval.")
        print()
        print("Query the system:")
        print("  python demo.py")
        print()
        print("Or use the API:")
        print("  python api.py")
        
    except Exception as e:
        print(f"\nERROR during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


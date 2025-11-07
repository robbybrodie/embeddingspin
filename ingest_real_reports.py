#!/usr/bin/env python3
"""
Ingest Real IBM Annual Reports
================================

This script ingests real IBM annual reports (2016-2024) into the temporal spin
retrieval system.

Prerequisites:
1. Download IBM 10-K reports from https://www.ibm.com/investor or SEC EDGAR
2. Place them in a directory (PDF or TXT format)
3. Name them with the year, e.g., "IBM_2016_10K.pdf"

Usage:
    # With LlamaStack embeddings (production)
    python ingest_real_reports.py --directory ./ibm_reports --use-llamastack
    
    # With mock embeddings (testing)
    python ingest_real_reports.py --directory ./ibm_reports
    
    # Clear existing data first
    python ingest_real_reports.py --clear-first --directory ./ibm_reports
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import re

from llamastack_client import LlamaStackEmbeddingClient, MockEmbeddingClient
from vector_store import InMemoryVectorStore, ChromaVectorStore
from ingestion import TemporalSpinIngestionPipeline


def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file.
    
    Requires PyPDF2 or pdfplumber:
        pip install PyPDF2
    """
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            # Extract first 50 pages (enough for most financial highlights)
            for page_num in range(min(50, len(reader.pages))):
                text += reader.pages[page_num].extract_text()
            return text
    except ImportError:
        print("ERROR: PyPDF2 not installed. Install with: pip install PyPDF2")
        return None
    except Exception as e:
        print(f"ERROR extracting text from {pdf_path}: {e}")
        return None


def extract_year_from_filename(filename):
    """Extract year from filename like 'IBM_2019_10K.pdf'"""
    match = re.search(r'20\d{2}', filename)
    if match:
        return int(match.group())
    return None


def load_reports_from_directory(directory):
    """
    Load all reports from a directory.
    
    Returns:
        List of (text, timestamp, doc_id, filepath) tuples
    """
    directory = Path(directory)
    if not directory.exists():
        print(f"ERROR: Directory not found: {directory}")
        return []
    
    reports = []
    
    # Look for PDF, TXT, HTML, and HTM files
    for filepath in sorted(directory.glob("*")):
        if filepath.suffix.lower() not in ['.pdf', '.txt', '.html', '.htm']:
            continue
        
        print(f"Processing: {filepath.name}")
        
        # Extract year from filename
        year = extract_year_from_filename(filepath.name)
        if not year:
            print(f"  WARNING: Could not extract year from {filepath.name}, skipping")
            continue
        
        # Load text
        if filepath.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(filepath)
        elif filepath.suffix.lower() in ['.html', '.htm']:
            # Parse HTML
            try:
                from bs4 import BeautifulSoup
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f, 'html.parser')
                    # Remove scripts and styles
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text(separator='\n')
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    text = '\n'.join(line for line in lines if line)
            except ImportError:
                print(f"  WARNING: BeautifulSoup not installed, reading as plain text")
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
        else:  # .txt
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        if not text:
            print(f"  WARNING: No text extracted from {filepath.name}, skipping")
            continue
        
        # Create timestamp (December 31 of the year)
        timestamp = datetime(year, 12, 31, tzinfo=timezone.utc)
        
        # Create doc ID
        doc_id = f"ibm-10k-{year}"
        
        reports.append((text, timestamp, doc_id, str(filepath)))
        print(f"  ✓ Loaded: {len(text)} chars, Year: {year}")
    
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Ingest real IBM annual reports into temporal spin system"
    )
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help="Directory containing IBM report files (PDF or TXT)"
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
    print("IBM ANNUAL REPORTS INGESTION")
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
        print("\nClearing existing data...")
        vector_store.clear()
        print("✓ Data cleared")
    
    # Load reports
    print(f"\nLoading reports from: {args.directory}")
    print("-"*80)
    reports = load_reports_from_directory(args.directory)
    
    if not reports:
        print("\nERROR: No reports found!")
        print("\nMake sure:")
        print("  1. The directory exists")
        print("  2. It contains PDF or TXT files")
        print("  3. Filenames include years (e.g., IBM_2019_10K.pdf)")
        sys.exit(1)
    
    print(f"\n✓ Found {len(reports)} reports")
    print()
    
    # Create ingestion pipeline
    pipeline = TemporalSpinIngestionPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store
    )
    
    # Ingest reports
    print("Ingesting reports (this may take a while with real embeddings)...")
    print("-"*80)
    
    texts = [text for text, _, _, _ in reports]
    timestamps = [ts for _, ts, _, _ in reports]
    doc_ids = [doc_id for _, _, doc_id, _ in reports]
    metadatas = [{"filepath": fp} for _, _, _, fp in reports]
    
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
        
        # Show summary
        print("Documents in vector store:")
        for doc in docs:
            year = doc.timestamp.year
            phi_deg = doc.phi * 180 / 3.14159
            print(f"  • {year}: Phase φ = {phi_deg:.1f}°, ID = {doc.doc_id}")
        
        print()
        print("You can now query the system:")
        print("  python demo.py --query 'IBM revenue 2019' --timestamp 2019-01-01 --beta 10")
        print()
        print("Or start the API server:")
        print("  python api.py")
        
    except Exception as e:
        print(f"\nERROR during ingestion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
Mock Dataset Generator for Temporal Spin Retrieval Demo
========================================================

Generates realistic mock financial reports for IBM spanning 2015-2024.
Each report includes:
- Unique timestamp (year-specific)
- Financial metrics with year-over-year variations
- Revenue, profit, and strategic initiative data
- Natural language suitable for semantic search
"""

from datetime import datetime, timezone
from typing import List, Tuple
import random


def generate_ibm_reports() -> List[Tuple[str, datetime]]:
    """
    Generate 10 mock IBM financial reports (2015-2024).
    
    Each report includes:
    - Clear date indicators ("for the period ended December 31, YYYY")
    - Revenue and profit figures that evolve over time
    - Strategic initiatives and technology focus areas
    - Natural language suitable for embedding
    
    Returns:
        List of (text, timestamp) tuples
    """
    
    # Base financial data with realistic year-over-year changes
    financial_data = [
        # (year, revenue_billions, net_income_billions, strategic_focus)
        (2015, 81.7, 13.2, "cloud computing transformation and Watson AI"),
        (2016, 79.9, 11.9, "cognitive solutions and cloud platform growth"),
        (2017, 79.1, 5.8, "strategic imperatives including cloud and analytics"),
        (2018, 79.6, 8.7, "hybrid cloud and AI-driven solutions"),
        (2019, 77.1, 9.4, "Red Hat acquisition and hybrid multi-cloud strategy"),
        (2020, 73.6, 5.6, "hybrid cloud platform and business automation"),
        (2021, 57.4, 5.7, "infrastructure modernization and application development"),
        (2022, 60.5, 1.6, "hybrid cloud and AI capabilities across industries"),
        (2023, 61.9, 7.5, "watsonx AI platform and enterprise automation"),
        (2024, 63.2, 8.1, "quantum computing and generative AI for enterprises"),
    ]
    
    reports = []
    
    for year, revenue, net_income, strategic_focus in financial_data:
        # Create timestamp (December 31 of each year)
        timestamp = datetime(year, 12, 31, tzinfo=timezone.utc)
        
        # Calculate growth rates
        prev_revenue = financial_data[financial_data.index((year, revenue, net_income, strategic_focus)) - 1][1] if year > 2015 else revenue
        revenue_growth = ((revenue - prev_revenue) / prev_revenue) * 100 if year > 2015 else 0
        
        # Generate report text
        text = f"""
IBM Corporation Annual Financial Report
For the period ended December 31, {year}

EXECUTIVE SUMMARY

IBM Corporation reported financial results for the fiscal year {year}, demonstrating 
continued execution of our strategic transformation focused on {strategic_focus}.

FINANCIAL HIGHLIGHTS

Total Revenue: ${revenue:.1f} billion
- Year-over-year change: {revenue_growth:+.1f}%
- Driven by strong performance in cloud and cognitive solutions

Net Income: ${net_income:.1f} billion
- Reflects ongoing investments in innovation and strategic positioning

KEY BUSINESS METRICS

Cloud Revenue: ${revenue * 0.25:.1f} billion (approximately 25% of total revenue)
Cognitive Solutions: Growing adoption across enterprise clients
Research & Development: Continued investment in emerging technologies

STRATEGIC INITIATIVES

In {year}, IBM focused on {strategic_focus}, delivering innovative 
solutions to our global client base. Our hybrid cloud platform continues to gain 
traction as enterprises modernize their IT infrastructure.

OUTLOOK

Looking ahead to {year + 1}, IBM remains committed to driving value through 
technological innovation and strategic partnerships. We are well-positioned to 
capitalize on growing demand for hybrid cloud, AI, and automation solutions.

SEGMENT PERFORMANCE

Software: Strong performance driven by {strategic_focus}
Consulting: Robust growth in cloud transformation and modernization projects
Infrastructure: Steady demand for hybrid cloud infrastructure solutions

Management believes IBM is well-positioned for sustainable long-term growth 
through continued focus on high-value segments and emerging technologies.

For more information, visit ibm.com/investor or contact IBM Investor Relations.
""".strip()
        
        reports.append((text, timestamp))
    
    return reports


def generate_query_examples() -> List[Tuple[str, datetime, str]]:
    """
    Generate example queries for demonstrating temporal spin retrieval.
    
    Returns:
        List of (query_text, query_timestamp, description) tuples
    """
    queries = [
        (
            "IBM revenue and financial performance",
            datetime(2016, 6, 30, tzinfo=timezone.utc),
            "Mid-2016 query - should prioritize 2016 report with temporal zoom"
        ),
        (
            "IBM cloud computing strategy and growth",
            datetime(2019, 12, 31, tzinfo=timezone.utc),
            "End of 2019 - Red Hat acquisition era, should favor 2019-2020"
        ),
        (
            "IBM artificial intelligence and Watson capabilities",
            datetime(2015, 12, 31, tzinfo=timezone.utc),
            "2015 query - Watson AI focus period"
        ),
        (
            "IBM hybrid cloud platform and enterprise solutions",
            datetime(2021, 6, 30, tzinfo=timezone.utc),
            "Mid-2021 - post-Red Hat integration, hybrid cloud emphasis"
        ),
        (
            "IBM quantum computing and generative AI",
            datetime(2024, 6, 30, tzinfo=timezone.utc),
            "Recent 2024 - quantum and gen AI focus"
        ),
    ]
    
    return queries


def print_dataset_summary(reports: List[Tuple[str, datetime]]) -> None:
    """
    Print a summary of the generated dataset.
    
    Args:
        reports: List of (text, timestamp) tuples
    """
    print("=" * 80)
    print("TEMPORAL SPIN RETRIEVAL - DEMO DATASET")
    print("=" * 80)
    print()
    print(f"Total Reports: {len(reports)}")
    print()
    print("Report Timestamps:")
    print("-" * 40)
    for i, (text, timestamp) in enumerate(reports, 1):
        # Extract year and revenue from text
        year = timestamp.year
        revenue_line = [line for line in text.split('\n') if 'Total Revenue:' in line][0]
        print(f"{i:2d}. {year} - {revenue_line.strip()}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    # Generate and display dataset
    reports = generate_ibm_reports()
    print_dataset_summary(reports)
    
    print("\nSample Report (2019):")
    print("-" * 80)
    sample_report = [r for r in reports if r[1].year == 2019][0]
    print(sample_report[0][:500] + "...\n")
    
    print("\nExample Queries:")
    print("-" * 80)
    queries = generate_query_examples()
    for i, (query, timestamp, description) in enumerate(queries, 1):
        print(f"{i}. Query: \"{query}\"")
        print(f"   Timestamp: {timestamp.date()}")
        print(f"   Expected: {description}")
        print()


"""
Mock Dataset Generator for Temporal Spin Retrieval Demo
========================================================

Generates realistic mock financial reports for IBM spanning 2015-2024.
Each report includes:
- A temporal interval (the fiscal year it covers)
- Financial metrics with year-over-year variations
- Revenue, profit, and strategic initiative data
- Natural language suitable for semantic search

An annual report is a *period*, not an instant. Encoding one as a point at
31 December collapses twelve months onto a single phase and makes it look, to a
Q1 query, like a December document rather than a document covering Q1. Prefer
:func:`generate_ibm_report_intervals`; :func:`generate_ibm_reports` is retained
for callers still working in point mode.

The set also includes one multi-year review document, which crosses 1-year period
boundaries and therefore demonstrates boundary splitting into several
representations sharing a ``group_id``.
"""

from datetime import datetime, timezone
from typing import List, Tuple

from temporal_encoding import TemporalInterval


def _build_reports() -> List[Tuple[str, int]]:
    """Generate the report texts paired with the fiscal year each one covers."""

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
    
    for index, (year, revenue, net_income, strategic_focus) in enumerate(financial_data):
        # Calculate growth rates
        prev_revenue = financial_data[index - 1][1] if index > 0 else revenue
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
        
        reports.append((text, year))

    return reports


def generate_ibm_report_intervals(
    include_multi_year: bool = True,
) -> List[Tuple[str, TemporalInterval]]:
    """
    Generate the IBM reports paired with the fiscal-year interval each covers.

    Args:
        include_multi_year: Append a 2017-2022 review document. It crosses five
            1-year boundaries, so ingestion splits it into six representations
            under one ``group_id`` — the boundary-splitting path in miniature.

    Returns:
        List of ``(text, TemporalInterval)`` tuples.
    """
    reports = [(text, TemporalInterval.of_year(year)) for text, year in _build_reports()]

    if include_multi_year:
        reports.append(
            (
                "IBM Corporation Six-Year Strategic Review, 2017-2022\n\n"
                "This review covers the period from 2017 through 2022, spanning the "
                "Red Hat acquisition, the Kyndryl separation, and the pivot to hybrid "
                "cloud and AI. Revenue moved from $79.1 billion in 2017 to $60.5 "
                "billion in 2022, reflecting the divestiture of the managed "
                "infrastructure business rather than a contraction in demand.",
                TemporalInterval.spanning(2017, 2022),
            )
        )

    return reports


def generate_ibm_reports() -> List[Tuple[str, datetime]]:
    """
    Point-mode view of the dataset, one timestamp per report.

    Kept for callers that predate interval encoding. New code should use
    :func:`generate_ibm_report_intervals` — a year-long arc is what lets an annual
    report answer a quarterly query.
    """
    return [
        (text, datetime(year, 12, 31, tzinfo=timezone.utc))
        for text, year in _build_reports()
    ]


def generate_query_examples() -> List[Tuple[str, TemporalInterval, str]]:
    """
    Example queries for demonstrating temporal spin retrieval.

    Returns:
        List of ``(query_text, interval, description)`` tuples.
    """
    return [
        (
            "IBM revenue and financial performance",
            TemporalInterval.of_quarter(2016, 2),
            "Q2 2016 - should prioritise the 2016 report; 2015 and 2017 coincide "
            "on the 1-year circle and are separated by the 16-year circle",
        ),
        (
            "IBM cloud computing strategy and growth",
            TemporalInterval.of_year(2019),
            "FY2019 - Red Hat acquisition era",
        ),
        (
            "IBM artificial intelligence and Watson capabilities",
            TemporalInterval.of_year(2015),
            "FY2015 - Watson AI focus period",
        ),
        (
            "IBM hybrid cloud platform and enterprise solutions",
            TemporalInterval.spanning(2020, 2022),
            "A three-year span - matches the multi-year review through several of "
            "its split representations, deduplicated back to one hit",
        ),
        (
            "IBM quantum computing and generative AI",
            TemporalInterval.of_quarter(2024, 3),
            "Q3 2024 - quantum and gen AI focus",
        ),
    ]


def generate_point_query_examples() -> List[Tuple[str, datetime, str]]:
    """Point-mode example queries, retained for the legacy demo path."""
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


def print_dataset_summary(reports: List[Tuple[str, TemporalInterval]]) -> None:
    """
    Print a summary of the generated dataset.

    Args:
        reports: List of ``(text, interval)`` tuples.
    """
    print("=" * 80)
    print("TEMPORAL SPIN RETRIEVAL - DEMO DATASET")
    print("=" * 80)
    print()
    print(f"Total Reports: {len(reports)}")
    print()
    print("Report Periods:")
    print("-" * 60)
    for i, (text, interval) in enumerate(reports, 1):
        revenue_lines = [line for line in text.split("\n") if "Total Revenue:" in line]
        detail = revenue_lines[0].strip() if revenue_lines else text.split("\n")[0].strip()
        end = interval.end.date() if interval.end else interval.start.date()
        print(f"{i:2d}. [{interval.start.date()} -> {end})  {detail}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    reports = generate_ibm_report_intervals()
    print_dataset_summary(reports)

    print("\nSample Report (2019):")
    print("-" * 80)
    sample = [r for r in reports if r[1].start.year == 2019][0]
    print(sample[0][:500] + "...\n")

    print("\nExample Queries:")
    print("-" * 80)
    for i, (query, interval, description) in enumerate(generate_query_examples(), 1):
        print(f'{i}. Query: "{query}"')
        print(f"   Period:   {interval}")
        print(f"   Expected: {description}")
        print()


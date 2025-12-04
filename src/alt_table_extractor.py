# src/alt_table_extractor.py
from __future__ import annotations

from typing import List
import pandas as pd
import pdfplumber

from src.table_extractor import ExtractedTable


def extract_tables_from_pdf(
    pdf_path: str,
    min_rows: int = 2,
) -> List[ExtractedTable]:
    """
    LAST-RESORT table fallback using pdfplumber directly on the PDF.

    - Runs only if Docling + JSON fallback found no tables.
    - Returns ExtractedTable objects with source='pdfplumber'.
    """
    tables: List[ExtractedTable] = []

    print(f"[ALT TABLE] Opening PDF with pdfplumber: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            print(f"[ALT TABLE] PDF has {num_pages} page(s)")

            # Two strategies: 'lines' then 'text'
            line_settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            }
            text_settings = {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
            }

            for page_num, page in enumerate(pdf.pages, start=1):
                print(f"[ALT TABLE] Page {page_num}: extracting tables (lines strategy)...")
                try:
                    raw_tables_lines = page.extract_tables(table_settings=line_settings)
                except Exception as e:
                    print(f"[ALT TABLE] Page {page_num} lines-strategy error: {e}")
                    raw_tables_lines = []

                print(f"[ALT TABLE] Page {page_num}: {len(raw_tables_lines)} table(s) via lines")

                # If lines strategy fails, also try text-based detection
                print(f"[ALT TABLE] Page {page_num}: extracting tables (text strategy)...")
                try:
                    raw_tables_text = page.extract_tables(table_settings=text_settings)
                except Exception as e:
                    print(f"[ALT TABLE] Page {page_num} text-strategy error: {e}")
                    raw_tables_text = []

                print(f"[ALT TABLE] Page {page_num}: {len(raw_tables_text)} table(s) via text")

                # Merge both sets
                all_raw = (raw_tables_lines or []) + (raw_tables_text or [])

                for idx, raw in enumerate(all_raw, start=1):
                    if not raw:
                        continue

                    df = pd.DataFrame(raw)

                    # Drop tables that are basically empty
                    if df.dropna(how="all").shape[0] < min_rows:
                        continue

                    tables.append(
                        ExtractedTable(
                            dataframe=df,
                            page=page_num,
                            caption=None,
                            source="pdfplumber",
                            table_id=f"pdfplumber_p{page_num}_{idx}",
                        )
                    )

    except Exception as e:
        print(f"[ALT TABLE] pdfplumber failed on {pdf_path}: {e}")
        return []

    print(f"[ALT TABLE] pdfplumber extracted {len(tables)} table(s) from {pdf_path}")
    return tables
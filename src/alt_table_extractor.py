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

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    raw_tables = page.extract_tables()
                except Exception:
                    continue

                if not raw_tables:
                    continue

                for idx, raw in enumerate(raw_tables, start=1):
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

    print(f"[ALT TABLE] pdfplumber extracted {len(tables)} tables from {pdf_path}")
    return tables
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import logging
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ExtractedTable:
    dataframe: pd.DataFrame
    page: Optional[int]
    caption: Optional[str]
    source: str  # "docling" or "fallback"
    table_id: str


def _bbox_coords(bbox) -> Optional[tuple]:
    """
    Normalize bbox into (x1, y1, x2, y2).

    Docling sometimes gives:
      - bbox as dict: {"l": ..., "t": ..., "r": ..., "b": ...}
      - or as list/tuple [x1, y1, x2, y2]
    """
    if bbox is None:
        return None

    # dict style
    if isinstance(bbox, dict):
        try:
            return float(bbox["l"]), float(bbox["t"]), float(bbox["r"]), float(bbox["b"])
        except Exception:
            return None

    # list/tuple style
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = bbox[:4]
            return float(x1), float(y1), float(x2), float(y2)
        except Exception:
            return None

    return None


def extract_tables(
    doc: Any,
    *,
    use_fallback_if_empty: bool = True,
    min_rows: int = 2,
    y_tolerance: float = 8.0,
) -> List[ExtractedTable]:
    """
    Unified table extraction for a DoclingDocument.

    1) Try Docling's native tables (doc.tables)
    2) If none, optionally run a position-based fallback using iterate_items().
    """
    tables: List[ExtractedTable] = []

    # ---------- 1) Native Docling tables ----------
    if hasattr(doc, "tables") and doc.tables:
        for i, table in enumerate(doc.tables, start=1):
            try:
                df: pd.DataFrame = table.export_to_dataframe(doc=doc)

                page_no = None
                prov = getattr(table, "prov", []) or []
                if prov:
                    page_no = getattr(prov[0], "page_no", None)

                caption = None
                if hasattr(table, "caption_text"):
                    try:
                        # most recent docling-core uses caption_text(doc)
                        caption = table.caption_text(doc)
                    except TypeError:
                        # older versions may expose caption_text as a plain property
                        try:
                            caption = table.caption_text
                        except Exception:
                            caption = None

                tables.append(
                    ExtractedTable(
                        dataframe=df,
                        page=page_no,
                        caption=caption,
                        source="docling",
                        table_id=f"table_{i}",
                    )
                )
            except Exception as e:
                log.warning("Could not export Docling table %s: %s", i, e)

    # If we got any native tables, we stop here – NO JSON fallback
    if tables or not use_fallback_if_empty:
        return tables

    # ---------- 2) Fallback: position-based from iterate_items ----------
    if not hasattr(doc, "iterate_items"):
        log.info("Docling doc has no iterate_items(); skipping fallback.")
        return []

    lines: List[Dict[str, Any]] = []

    for node, _ in doc.iterate_items():
        data = node.model_dump()  # pydantic v2

        text = (data.get("text") or "").strip()
        if not text:
            continue

        prov = data.get("prov") or []
        if not prov:
            continue

        bbox_data = prov[0].get("bbox")
        page_no = prov[0].get("page_no")

        coords = _bbox_coords(bbox_data)
        if coords is None:
            continue

        x1, y1, _, _ = coords

        lines.append(
            {
                "text": text,
                "x": float(x1),
                "y": float(y1),
                "page": page_no,
            }
        )

    if not lines:
        log.info("Fallback found no positioned text.")
        return []

    pages = sorted({ln["page"] for ln in lines})
    fallback_tables: List[ExtractedTable] = []

    for page in pages:
        page_lines = [ln for ln in lines if ln["page"] == page]

        # cluster by approximate y, then sort by x
        page_lines.sort(key=lambda d: (round(d["y"] / y_tolerance), d["x"]))

        rows: List[List[str]] = []
        current_row: List[str] = []
        current_y: Optional[float] = None

        for ln in page_lines:
            if current_y is None or abs(ln["y"] - current_y) <= y_tolerance:
                current_row.append(ln["text"])
                if current_y is None:
                    current_y = ln["y"]
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [ln["text"]]
                current_y = ln["y"]

        if current_row:
            rows.append(current_row)

        if len(rows) < min_rows:
            continue

        max_len = max(len(r) for r in rows)
        norm_rows = [r + [""] * (max_len - len(r)) for r in rows]
        df = pd.DataFrame(norm_rows)

        fallback_tables.append(
            ExtractedTable(
                dataframe=df,
                page=page,
                caption=None,
                source="fallback",
                table_id=f"fallback_page_{page}",
            )
        )

    return fallback_tables

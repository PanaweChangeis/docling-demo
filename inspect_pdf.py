from __future__ import annotations

import pdfplumber
from pypdf import PdfReader


def inspect_pdf_nature(pdf_path: str) -> None:
    print(f"=== Inspecting {pdf_path} ===")

    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    print(f"Pages: {num_pages}")

    total_text_chars = 0
    example_page_text = ""

    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        total_text_chars += len(txt)
        if i == 0:
            example_page_text = txt[:500]

    print(f"Total text chars (pypdf): {total_text_chars}")
    print("Sample from page 1 (pypdf):")
    print(repr(example_page_text))

    num_pages_with_text_objs = 0
    num_pages_with_images = 0

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            objs = page.objects
            text_objs = objs.get("text", [])
            image_objs = objs.get("image", [])
            line_objs = objs.get("line", [])

            if text_objs:
                num_pages_with_text_objs += 1
            if image_objs:
                num_pages_with_images += 1

            if i == 1:
                print(
                    f"Page 1: text_objs={len(text_objs)}, "
                    f"images={len(image_objs)}, lines={len(line_objs)}"
                )

    print(f"Pages with text objects: {num_pages_with_text_objs}/{num_pages}")
    print(f"Pages with image objects: {num_pages_with_images}/{num_pages}")

    if (
        total_text_chars == 0
        and num_pages_with_text_objs == 0
        and num_pages_with_images > 0
    ):
        nature = "IMAGE_ONLY"
    elif total_text_chars > 0 and num_pages_with_text_objs > 0:
        nature = "TEXT_BASED"
    elif num_pages_with_text_objs > 0 and num_pages_with_images > 0:
        nature = "HYBRID"
    else:
        nature = "WEIRD/ENCODED"

    print(f"🧬 PDF nature: {nature}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to the PDF to inspect")
    args = parser.parse_args()

    inspect_pdf_nature(args.pdf_path)

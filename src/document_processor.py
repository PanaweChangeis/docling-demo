# # """
# # Docling integration for processing uploaded documents.
# # """

# # import os
# # import tempfile
# # from typing import List, Any
# # from pathlib import Path
# # from docling.document_converter import DocumentConverter, PdfFormatOption
# # from docling.datamodel.base_models import InputFormat
# # from docling.datamodel.pipeline_options import PdfPipelineOptions
# # from langchain_core.documents import Document


# # class DocumentProcessor:
# #     """Handles document processing using Docling."""

# #     def __init__(self):
# #         """Initialize the Docling DocumentConverter."""
# #         # Configure pipeline options for PDF processing
# #         pipeline_options = PdfPipelineOptions()
# #         pipeline_options.do_ocr = True
# #         pipeline_options.do_table_structure = True
# #         pipeline_options.generate_picture_images = True  # Enable image extraction
# #         pipeline_options.images_scale = 2.0  # Higher resolution for better quality

# #         # Initialize converter with PDF options
# #         self.converter = DocumentConverter(
# #             format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
# #         )

# #     def process_uploaded_files(self, uploaded_files) -> tuple[List[Document], List[Any]]:
# #         """
# #         Process uploaded files and convert them to LangChain Document objects.

# #         Args:
# #             uploaded_files: List of Streamlit UploadedFile objects

# #         Returns:
# #             Tuple of (LangChain Documents, Docling Documents)
# #         """
# #         documents = []
# #         docling_docs = []
# #         temp_dir = tempfile.mkdtemp()

# #         try:
# #             for uploaded_file in uploaded_files:
# #                 print(f"📄 Processing {uploaded_file.name}...")

# #                 # Save uploaded file to temporary location
# #                 temp_file_path = os.path.join(temp_dir, uploaded_file.name)
# #                 with open(temp_file_path, "wb") as f:
# #                     f.write(uploaded_file.getbuffer())

# #                 # Process the document with Docling
# #                 try:
# #                     result = self.converter.convert(temp_file_path)

# #                     # Export to markdown
# #                     markdown_content = result.document.export_to_markdown()

# #                     # Create LangChain document
# #                     doc = Document(
# #                         page_content=markdown_content,
# #                         metadata={
# #                             "filename": uploaded_file.name,
# #                             "file_type": uploaded_file.type,
# #                             "source": uploaded_file.name,
# #                         },
# #                     )
# #                     documents.append(doc)

# #                     # Store the Docling document for structure visualization
# #                     docling_docs.append({
# #                         'filename': uploaded_file.name,
# #                         'doc': result.document
# #                     })

# #                     print(f"✅ Successfully processed {uploaded_file.name}")

# #                 except Exception as e:
# #                     print(f"❌ Error processing {uploaded_file.name}: {str(e)}")
# #                     continue

# #         finally:
# #             # Clean up temporary files
# #             try:
# #                 import shutil

# #                 shutil.rmtree(temp_dir)
# #             except Exception as e:
# #                 print(f"⚠️ Warning: Could not clean up temp directory: {str(e)}")

# #         print(f"✅ Processed {len(documents)} documents successfully")
# #         return documents, docling_docs

# """
# Docling integration for processing uploaded documents.
# """

# import os
# from typing import List, Any
# from pathlib import Path
# from pypdf import PdfReader
# import pdfplumber

# from pdf2image import convert_from_path
# import pytesseract
# import numpy as np
# from paddleocr import PaddleOCR

# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from langchain_core.documents import Document


# class DocumentProcessor:
#     def __init__(self, force_ocr: bool = False):
#         """
#         Initialize the Docling DocumentConverter and output directory.

#         """
#         self.force_ocr = force_ocr

#         # Configure pipeline options for PDF processing
#         pipeline_options = PdfPipelineOptions(
#             do_ocr=self.force_ocr,   
#             do_table_structure=True,
#             generate_picture_images=True,
#             images_scale=2.0 if self.force_ocr else 1.0,
#         )

#         # Initialize converter with PDF options
#         self.converter = DocumentConverter(
#             format_options={
#                 InputFormat.PDF: PdfFormatOption(
#                     pipeline_options=pipeline_options
#                 )
#             }
#         )
#         # Where we will store original files + markdown + json
#         self.output_root = Path("outputs")
#         self.output_root.mkdir(exist_ok=True)

#         # Initialize PaddleOCR only when we may need aggressive OCR
#         self.paddle_ocr = None
#         if self.force_ocr:
#             print("🔧 Initializing PaddleOCR (this may take a bit on first run)...")
#             self.paddle_ocr = PaddleOCR(lang="en", use_angle_cls=True)

        
#     def _ocr_pdf_with_paddleocr(
#         self,
#         file_path: Path,
#         dpi: int = 200,
#         max_pages: int = 5,
#     ) -> str:
#         """
#         Fallback OCR: render up to `max_pages` PDF pages to images and run PaddleOCR.
#         Returns concatenated text for the processed pages.
#         """
#         if self.paddle_ocr is None:
#             # Lazy init in case force_ocr was toggled later
#             print("🔧 Lazy-initializing PaddleOCR...")
#             self.paddle_ocr = PaddleOCR(lang="en", use_angle_cls=True)

#         print(
#             f"🔍 PaddleOCR fallback on {file_path} "
#             f"(dpi={dpi}, max_pages={max_pages}) ..."
#         )

#         try:
#             pages = convert_from_path(
#                 str(file_path),
#                 dpi=dpi,
#                 first_page=1,
#                 last_page=max_pages,
#             )
#         except Exception as e:
#             print(f"❌ pdf2image convert_from_path failed: {e}")
#             return ""

#         texts: List[str] = []
#         total_pages = len(pages)
#         print(f"📝 Rendering {total_pages} page(s) for OCR...")

#         for i, page in enumerate(pages, start=1):
#             print(f"   🧠 PaddleOCR on page {i}/{total_pages} ...")
#             try:
#                 img = np.array(page)  # pdf2image gives PIL.Image; convert to numpy
#                 result = self.paddle_ocr.ocr(img, cls=True)

#                 # result is a list; we take the first page's lines
#                 page_lines = []
#                 if result:
#                     for line in result[0]:
#                         text = line[1][0]
#                         if text:
#                             page_lines.append(text)

#                 page_text = "\n".join(page_lines)
#                 if page_text.strip():
#                     texts.append(page_text)
#             except Exception as e:
#                 print(f"❌ PaddleOCR failed on page {i}: {e}")

#         full_text = "\n\n".join(texts)
#         print(f"✅ PaddleOCR extracted {len(full_text)} characters")
#         return full_text


#     def process_uploaded_files(self, uploaded_files) -> tuple[List[Document], List[Any]]:
#         """
#         Process uploaded files and convert them to LangChain Document objects.

#         Args:
#             uploaded_files: List of Streamlit UploadedFile objects

#         Returns:
#             Tuple of (LangChain Documents, Docling Documents)
#         """
#         documents: List[Document] = []
#         docling_docs: List[Any] = []

#         for uploaded_file in uploaded_files:
#             filename = uploaded_file.name
#             print(f"📄 Processing {filename}...")

#             # Create per-document output folder: outputs/<file-stem>/
#             doc_dir = self.output_root / Path(filename).stem
#             doc_dir.mkdir(parents=True, exist_ok=True)

#             # 1) Save original uploaded file
#             original_path = doc_dir / filename
#             with open(original_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())

#             # 2) Process the document with Docling
#             try:
#                 result = self.converter.convert(str(original_path))
#                 dl_doc = result.document

#                 # 3) Export to markdown and save as document.md
#                 markdown_content = dl_doc.export_to_markdown()

#                 # if self.force_ocr and len(markdown_content.strip()) < 50:
#                 #     print("⚠️ Docling markdown is very short; running PaddleOCR fallback...")
#                 #     ocr_text = self._ocr_pdf_with_paddleocr(
#                 #         original_path,
#                 #         dpi=200,
#                 #         max_pages=5,   # you can increase later if this works well
#                 #     )
#                 #     if ocr_text.strip():
#                 #         markdown_content = ocr_text
#                 #         (doc_dir / "ocr_document.txt").write_text(
#                 #             ocr_text, encoding="utf-8"
#                 #         )
#                 #     else:
#                 #         print("❌ PaddleOCR fallback produced no text; keeping Docling output.")

#                 (doc_dir / "document.md").write_text(markdown_content, encoding="utf-8")

#                 # 4) Try to export full schema as JSON (best-effort)
#                 try:
#                     # Docling docs are Pydantic models (v2 style)
#                     json_str = dl_doc.model_dump_json(indent=2)
#                     (doc_dir / "document.json").write_text(json_str, encoding="utf-8")
#                 except Exception as e:
#                     print(f"⚠️ Could not save JSON schema for {filename}: {e}")

#                 # 5) Create LangChain document for RAG
#                 doc = Document(
#                     page_content=markdown_content,
#                     metadata={
#                         "filename": filename,
#                         "file_type": uploaded_file.type,
#                         "source": filename,
#                         "output_dir": str(doc_dir),
#                     },
#                 )
#                 documents.append(doc)

#                 # 6) Keep Docling document for structure visualizer
#                 docling_docs.append({
#                     'filename': filename,
#                     'doc': dl_doc,
#                     'pdf_path': str(original_path),
#                 })


#                 print(f"✅ Successfully processed {filename}")
#                 print(f"   → Original:   {original_path}")
#                 print(f"   → Markdown:   {doc_dir / 'document.md'}")
#                 print(f"   → JSON schema:{doc_dir / 'document.json'}")

#             except Exception as e:
#                 print(f"❌ Error processing {filename}: {str(e)}")
#                 continue

#         print(f"✅ Processed {len(documents)} documents successfully")
#         return documents, docling_docs

"""
Docling integration for processing uploaded documents.
"""

import os
from typing import List, Any
from pathlib import Path

from pypdf import PdfReader
import pdfplumber

from pdf2image import convert_from_path
import pytesseract
import numpy as np
from paddleocr import PaddleOCR

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_core.documents import Document


# ---------- PDF NATURE CLASSIFICATION (LIGHTWEIGHT) ----------

def classify_pdf_nature(pdf_path: str, sample_pages: int = 3) -> str:
    """
    Classify a PDF as TEXT_BASED, IMAGE_ONLY, HYBRID, or WEIRD_ENCODED.

    - Uses pypdf to measure total extractable text characters.
    - Uses pdfplumber on the first few pages to see if there are text/image objects.
    """
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
    except Exception as e:
        print(f"[NATURE] Could not open with PdfReader: {e}")
        return "UNKNOWN"

    total_text_chars = 0
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        total_text_chars += len(txt)

    pages_with_text_objs = 0
    pages_with_images = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                if i > sample_pages:
                    break
                objs = page.objects
                text_objs = objs.get("text", [])
                image_objs = objs.get("image", [])

                if text_objs:
                    pages_with_text_objs += 1
                if image_objs:
                    pages_with_images += 1
    except Exception as e:
        print(f"[NATURE] pdfplumber inspection failed: {e}")

    # Same logic as your inspector, but lighter
    if (
        total_text_chars == 0
        and pages_with_text_objs == 0
        and pages_with_images > 0
    ):
        nature = "IMAGE_ONLY"
    elif total_text_chars > 0 and pages_with_text_objs > 0:
        nature = "TEXT_BASED"
    elif pages_with_text_objs > 0 and pages_with_images > 0:
        nature = "HYBRID"
    elif total_text_chars > 0:
        # Text exists but no text objects/images in layout → your “weird” case
        nature = "WEIRD_ENCODED"
    else:
        nature = "UNKNOWN"

    print(
        f"[NATURE] {pdf_path} → {nature} "
        f"(chars={total_text_chars}, text_objs_pages={pages_with_text_objs}, "
        f"image_pages={pages_with_images})"
    )
    return nature


def pypdf_to_markdown(pdf_path: str) -> str:
    """
    Fallback: build a simple markdown string from pypdf text only.

    This is for WEIRD_ENCODED PDFs where Docling has no structured view,
    but pypdf can still read the text cleanly.
    """
    reader = PdfReader(pdf_path)
    chunks: List[str] = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            chunks.append(f"# Page {i}\n\n{txt}")

    md = "\n\n".join(chunks)
    print(f"[PYPDF] Built markdown from {len(reader.pages)} pages, {len(md)} chars")
    return md


class DocumentProcessor:
    def __init__(self, force_ocr: bool = False):
        """
        Initialize the Docling DocumentConverter and output directory.
        """
        self.force_ocr = force_ocr

        # Configure pipeline options for PDF processing
        pipeline_options = PdfPipelineOptions(
            do_ocr=self.force_ocr,
            do_table_structure=True,
            generate_picture_images=True,
            images_scale=2.0 if self.force_ocr else 1.0,
        )

        # Initialize converter with PDF options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        # Where we will store original files + markdown + json
        self.output_root = Path("outputs")
        self.output_root.mkdir(exist_ok=True)

        # Initialize PaddleOCR only when we may need aggressive OCR
        self.paddle_ocr = None
        if self.force_ocr:
            print("🔧 Initializing PaddleOCR (this may take a bit on first run)...")
            self.paddle_ocr = PaddleOCR(lang="en", use_angle_cls=True)

    def _ocr_pdf_with_paddleocr(
        self,
        file_path: Path,
        dpi: int = 200,
        max_pages: int = 5,
    ) -> str:
        """
        Fallback OCR: render up to `max_pages` PDF pages to images and run PaddleOCR.
        Returns concatenated text for the processed pages.
        """
        if self.paddle_ocr is None:
            # Lazy init in case force_ocr was toggled later
            print("🔧 Lazy-initializing PaddleOCR...")
            self.paddle_ocr = PaddleOCR(lang="en", use_angle_cls=True)

        print(
            f"🔍 PaddleOCR fallback on {file_path} "
            f"(dpi={dpi}, max_pages={max_pages}) ..."
        )

        try:
            pages = convert_from_path(
                str(file_path),
                dpi=dpi,
                first_page=1,
                last_page=max_pages,
            )
        except Exception as e:
            print(f"❌ pdf2image convert_from_path failed: {e}")
            return ""

        texts: List[str] = []
        total_pages = len(pages)
        print(f"📝 Rendering {total_pages} page(s) for OCR...")

        for i, page in enumerate(pages, start=1):
            print(f"   🧠 PaddleOCR on page {i}/{total_pages} ...")
            try:
                img = np.array(page)  # pdf2image gives PIL.Image; convert to numpy
                result = self.paddle_ocr.ocr(img, cls=True)

                page_lines = []
                if result:
                    for line in result[0]:
                        text = line[1][0]
                        if text:
                            page_lines.append(text)

                page_text = "\n".join(page_lines)
                if page_text.strip():
                    texts.append(page_text)
            except Exception as e:
                print(f"❌ PaddleOCR failed on page {i}: {e}")

        full_text = "\n\n".join(texts)
        print(f"✅ PaddleOCR extracted {len(full_text)} characters")
        return full_text

    def process_uploaded_files(self, uploaded_files) -> tuple[List[Document], List[Any]]:
        """
        Process uploaded files and convert them to LangChain Document objects.

        Args:
            uploaded_files: List of Streamlit UploadedFile objects

        Returns:
            Tuple of (LangChain Documents, Docling Documents)
        """
        documents: List[Document] = []
        docling_docs: List[Any] = []

        for uploaded_file in uploaded_files:
            filename = uploaded_file.name
            print(f"📄 Processing {filename}...")

            # Create per-document output folder: outputs/<file-stem>/
            doc_dir = self.output_root / Path(filename).stem
            doc_dir.mkdir(parents=True, exist_ok=True)

            # 1) Save original uploaded file
            original_path = doc_dir / filename
            with open(original_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            suffix = original_path.suffix.lower()
            is_pdf = suffix == ".pdf"

            try:
                dl_doc = None
                markdown_content = ""

                if is_pdf:
                    # ---------- NATURE-BASED ROUTING FOR PDFs ----------
                    nature = classify_pdf_nature(str(original_path))
                    print(f"[ROUTING] {filename} classified as {nature}")

                    if nature == "WEIRD_ENCODED":
                        # 🚧 Skip Docling, use pypdf-only markdown for RAG
                        markdown_content = pypdf_to_markdown(str(original_path))
                        # No Docling doc -> no tables/structure for this file
                    else:
                        # Normal path: let Docling handle layout, tables, etc.
                        result = self.converter.convert(str(original_path))
                        dl_doc = result.document
                        markdown_content = dl_doc.export_to_markdown()

                        # Optional OCR fallback (still disabled for stability)
                        # if self.force_ocr and len(markdown_content.strip()) < 50:
                        #     print("⚠️ Docling markdown is very short; running PaddleOCR fallback...")
                        #     ocr_text = self._ocr_pdf_with_paddleocr(
                        #         original_path,
                        #         dpi=200,
                        #         max_pages=5,
                        #     )
                        #     if ocr_text.strip():
                        #         markdown_content = ocr_text
                        #         (doc_dir / "ocr_document.txt").write_text(
                        #             ocr_text, encoding="utf-8"
                        #         )
                        #     else:
                        #         print("❌ PaddleOCR fallback produced no text; keeping Docling output.")
                else:
                    # Non-PDF (docx, pptx, html, etc.): just use Docling as before
                    result = self.converter.convert(str(original_path))
                    dl_doc = result.document
                    markdown_content = dl_doc.export_to_markdown()

                # 3) Save markdown
                (doc_dir / "document.md").write_text(markdown_content, encoding="utf-8")

                # 4) Save Docling JSON schema only if we have a Docling doc
                if dl_doc is not None:
                    try:
                        json_str = dl_doc.model_dump_json(indent=2)
                        (doc_dir / "document.json").write_text(json_str, encoding="utf-8")
                    except Exception as e:
                        print(f"⚠️ Could not save JSON schema for {filename}: {e}")

                # 5) Create LangChain document for RAG
                doc = Document(
                    page_content=markdown_content,
                    metadata={
                        "filename": filename,
                        "file_type": uploaded_file.type,
                        "source": filename,
                        "output_dir": str(doc_dir),
                        "extraction_mode": (
                            "pypdf" if is_pdf and dl_doc is None else "docling"
                        ),
                    },
                )
                documents.append(doc)

                # 6) Keep Docling document for structure visualizer only if available
                if dl_doc is not None:
                    docling_docs.append(
                        {
                            "filename": filename,
                            "doc": dl_doc,
                            "pdf_path": str(original_path),
                        }
                    )

                print(f"✅ Successfully processed {filename}")
                print(f"   → Original:   {original_path}")
                print(f"   → Markdown:   {doc_dir / 'document.md'}")
                if dl_doc is not None:
                    print(f"   → JSON schema:{doc_dir / 'document.json'}")
                else:
                    print("   → No Docling schema (pypdf-only extraction)")

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                continue

        print(f"✅ Processed {len(documents)} documents successfully")
        return documents, docling_docs

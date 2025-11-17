# """
# Docling integration for processing uploaded documents.
# """

# import os
# import tempfile
# from typing import List, Any
# from pathlib import Path
# from docling.document_converter import DocumentConverter, PdfFormatOption
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from langchain_core.documents import Document


# class DocumentProcessor:
#     """Handles document processing using Docling."""

#     def __init__(self):
#         """Initialize the Docling DocumentConverter."""
#         # Configure pipeline options for PDF processing
#         pipeline_options = PdfPipelineOptions()
#         pipeline_options.do_ocr = True
#         pipeline_options.do_table_structure = True
#         pipeline_options.generate_picture_images = True  # Enable image extraction
#         pipeline_options.images_scale = 2.0  # Higher resolution for better quality

#         # Initialize converter with PDF options
#         self.converter = DocumentConverter(
#             format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
#         )

#     def process_uploaded_files(self, uploaded_files) -> tuple[List[Document], List[Any]]:
#         """
#         Process uploaded files and convert them to LangChain Document objects.

#         Args:
#             uploaded_files: List of Streamlit UploadedFile objects

#         Returns:
#             Tuple of (LangChain Documents, Docling Documents)
#         """
#         documents = []
#         docling_docs = []
#         temp_dir = tempfile.mkdtemp()

#         try:
#             for uploaded_file in uploaded_files:
#                 print(f"📄 Processing {uploaded_file.name}...")

#                 # Save uploaded file to temporary location
#                 temp_file_path = os.path.join(temp_dir, uploaded_file.name)
#                 with open(temp_file_path, "wb") as f:
#                     f.write(uploaded_file.getbuffer())

#                 # Process the document with Docling
#                 try:
#                     result = self.converter.convert(temp_file_path)

#                     # Export to markdown
#                     markdown_content = result.document.export_to_markdown()

#                     # Create LangChain document
#                     doc = Document(
#                         page_content=markdown_content,
#                         metadata={
#                             "filename": uploaded_file.name,
#                             "file_type": uploaded_file.type,
#                             "source": uploaded_file.name,
#                         },
#                     )
#                     documents.append(doc)

#                     # Store the Docling document for structure visualization
#                     docling_docs.append({
#                         'filename': uploaded_file.name,
#                         'doc': result.document
#                     })

#                     print(f"✅ Successfully processed {uploaded_file.name}")

#                 except Exception as e:
#                     print(f"❌ Error processing {uploaded_file.name}: {str(e)}")
#                     continue

#         finally:
#             # Clean up temporary files
#             try:
#                 import shutil

#                 shutil.rmtree(temp_dir)
#             except Exception as e:
#                 print(f"⚠️ Warning: Could not clean up temp directory: {str(e)}")

#         print(f"✅ Processed {len(documents)} documents successfully")
#         return documents, docling_docs

"""
Docling integration for processing uploaded documents.
"""

import os
from typing import List, Any
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_core.documents import Document


class DocumentProcessor:
    """Handles document processing using Docling."""

    def __init__(self):
        """Initialize the Docling DocumentConverter and output directory."""
        # Configure pipeline options for PDF processing
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True  # Enable image extraction
        pipeline_options.images_scale = 1.0  # Higher resolution for better quality

        # Initialize converter with PDF options
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        # Where we will store original files + markdown + json
        self.output_root = Path("outputs")
        self.output_root.mkdir(exist_ok=True)

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

            # 2) Process the document with Docling
            try:
                result = self.converter.convert(str(original_path))
                dl_doc = result.document

                # 3) Export to markdown and save as document.md
                markdown_content = dl_doc.export_to_markdown()
                (doc_dir / "document.md").write_text(markdown_content, encoding="utf-8")

                # 4) Try to export full schema as JSON (best-effort)
                try:
                    # Docling docs are Pydantic models (v2 style)
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
                    },
                )
                documents.append(doc)

                # 6) Keep Docling document for structure visualizer
                docling_docs.append({"filename": filename, "doc": dl_doc})

                print(f"✅ Successfully processed {filename}")
                print(f"   → Original:   {original_path}")
                print(f"   → Markdown:   {doc_dir / 'document.md'}")
                print(f"   → JSON schema:{doc_dir / 'document.json'}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                continue

        print(f"✅ Processed {len(documents)} documents successfully")
        return documents, docling_docs

import hashlib
import os

from fastapi import HTTPException

from src.utils import config
from src.utils.api.task_registry import update_task
from src.utils.errors import EmbeddingError, MilvusError, PDFParseError
from src.utils.services.logger_config import logger
from src.utils.services.milvus_store import MilvusStoreHandler
from src.utils.services.pdf_parser import PDFParser


def _page_doc_id(pdf_name: str, page_idx: int) -> str:
    """Deterministic doc_id per (pdf, page).

    A fresh uuid4 per page made re-runs of a failed ingest create
    duplicates in Milvus. Hashing the pdf+page gives the same id on
    every run, so combined with delete-before-insert in
    `store_in_milvus`, re-ingesting a PDF is idempotent.
    """
    return hashlib.sha1(f"{pdf_name}:{page_idx}".encode()).hexdigest()[:16]


async def upload_pdf(
    pdf_name: str,
    collection_name: str,
    task_id: str,
):
    """Parse a PDF and index all pages into Milvus."""
    vector_store = MilvusStoreHandler(collection_name=collection_name)
    pdf_path = os.path.join(config.PDF_DIR, pdf_name)

    try:
        parser = PDFParser(pdf_path)
        pages = parser.parse_pdf()
        logger.info("Total pages parsed from PDF '%s': %d", pdf_path, len(pages))

        for i in range(len(pages)):
            long_text = pages[i].get("text")
            vector_store.store_in_milvus(
                text=long_text,
                doc_id=_page_doc_id(pdf_name, i),
                source=pdf_name,
            )
            logger.info("Uploaded page %d to the vector DB", i)
            update_task(task_id=task_id, status=f"uploaded page: {i} to db")
        update_task(task_id=task_id, status="Complete")
    except PDFParseError as e:
        logger.error("PDF parsing error for '%s': %s", pdf_path, e)
        raise HTTPException(
            status_code=400,
            detail={
                "error_type": "PDF_PARSE_ERROR",
                "message": str(e),
            },
        )
    except (EmbeddingError, MilvusError) as e:
        logger.exception("Vector store/embedding error during /upload_pdf: %s", e)
        raise HTTPException(
            status_code=502,
            detail={
                "error_type": "VECTOR_STORE_ERROR",
                "message": str(e),
            },
        )
    except Exception as e:
        logger.exception("Unexpected error during /upload_pdf: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Unexpected error while uploading PDF.",
        )

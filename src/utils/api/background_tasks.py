import os

from src.utils import config
from src.utils.services.milvus_store import MilvusStoreHandler
from src.utils.services.pdf_parser import PDFParser
from src.utils.services.logger_config import logger
from src.utils.errors import PDFParseError, EmbeddingError, MilvusError
from src.utils.api.task_registry import update_task
from fastapi import HTTPException



async def upload_pdf(pdf_name:str, collection_name:str, task_id):
    vector_store = MilvusStoreHandler(collection_name=collection_name)
    pdf_path = os.path.join(config.PDF_DIR, pdf_name)

    try:
        parser = PDFParser(pdf_path)
        pages = parser.parse_pdf()
        logger.info("Total pages parsed from PDF '%s': %d", pdf_path, len(pages))

        for i in range(1, len(pages)):
            long_text = pages[i].get("text")
            vector_store.store_in_milvus(text=long_text)
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

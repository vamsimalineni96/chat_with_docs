from openai import OpenAI
from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import NVIDIA_BASE_URL, NVIDIA_API_KEY, EMBED_MODEL
from src.utils.errors import EmbeddingError
from src.utils.services.logger_config import logger


class EmbeddingHandler:
    def __init__(self):
        logger.info("Initializing EmbeddingHandler...")

    def get_embedding(self, text: str, input_type: str = "query"):
        """
        Get a single embedding vector.

        Raises:
            EmbeddingError: for any upstream or unexpected failure.
        """
        try:
            client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

            response = client.embeddings.create(
                input=[text],
                model=EMBED_MODEL,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "NONE"},
            )

            embedding = response.data[0].embedding
            logger.info("Successfully retrieved embedding of length %d", len(embedding))
            return embedding

        except AuthenticationError as e:
            logger.error("Authentication with NVIDIA/OpenAI endpoint failed: %s", e)
            raise EmbeddingError(
                "Authentication with NVIDIA/OpenAI endpoint failed while creating embedding."
            ) from e

        except BadRequestError as e:
            logger.error("Bad request when creating embedding: %s", e)
            logger.debug("Offending text (first 200 chars): %r", text[:200])
            raise EmbeddingError("Bad request while creating embedding.") from e

        except RateLimitError as e:
            logger.warning("Rate limit hit while creating embedding: %s", e)
            raise EmbeddingError("Rate limit hit while creating embedding.") from e

        except APIConnectionError as e:
            logger.error("Connection error while calling embeddings API: %s", e)
            raise EmbeddingError("Connection error calling embeddings API.") from e

        except APIError as e:
            status = getattr(e, "status_code", None)
            logger.error(
                "APIError from embeddings endpoint (status=%s): %s", status, e
            )
            raise EmbeddingError("Upstream embedding API error.") from e

        except Exception as e:
            logger.exception("Unexpected error while creating embedding: %s", e)
            raise EmbeddingError("Unexpected error while creating embedding.") from e

    def get_document_embeddings(
        self, chunk_size: int, chunk_overlap: int, long_text: str
    ):
        """
        Splits text into chunks and retrieves embeddings for each chunk.

        Raises:
            EmbeddingError: if any chunk embedding fails.
        """
        logger.info(
            "Splitting text into chunks with size %d and overlap %d",
            chunk_size,
            chunk_overlap,
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?", " "],
        )
        text_chunks = splitter.split_text(long_text)

        logger.info("Generated %d text chunks", len(text_chunks))
        embeddings = []

        for i, chunk in enumerate(text_chunks):
            logger.info("Creating embedding for chunk %d/%d", i + 1, len(text_chunks))
            embedding = self.get_embedding(chunk, input_type="passage")
            embeddings.append(embedding)

        logger.info("Completed generating embeddings for document.")
        return embeddings, text_chunks

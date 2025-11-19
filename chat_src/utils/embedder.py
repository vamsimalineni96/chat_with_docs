import requests
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chat_src.utils.logger_config import LoggerConfig

# Initialize the logger
logger_config = LoggerConfig()
logger = logger_config.logger


class EmbeddingHandler:
    def __init__(self):

        self.NVIDIA_EMBEDDING_ENDPOINT = os.getenv("NVIDIA_EMBEDDING_ENDPOINT")
        self.NVIDIA_EMBEDDING_MODEL = os.getenv("NVIDIA_EMBEDDING_MODEL")
        self.NVIDIA_NIM_API_KEY = os.getenv("NVIDIA_NIM_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.NVIDIA_NIM_API_KEY}",
            "Content-Type": "application/json",
        }

    def get_embedding(self, text: str):
        """Fetches embeddings for the given text from NVIDIA NIM embedding model."""
        payload = {"model": self.NVIDIA_EMBEDDING_MODEL, "input": text}

        try:
            response = requests.post(
                self.NVIDIA_EMBEDDING_ENDPOINT, headers=self.headers, json=payload
            )
            response.raise_for_status()  # Raises HTTP errors if any

            json_response = json.loads(response.text)
            embedding = json_response.get("data", [{}])[0].get("embedding", [])

            if not embedding:
                logger.warning("Received empty embedding for input text.")
            else:
                logger.info("Successfully retrieved embedding.")

            return embedding

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error while fetching embedding: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding error while fetching embedding: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error while fetching embedding: {e}")
            return []

    def get_summary_embeddings(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 50
    ):
        """Splits the document into chunks and gets embeddings with exception handling."""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            text_chunks = text_splitter.split_text(text)
            logger.info(f"Successfully split document into {len(text_chunks)} chunks.")
        except Exception as e:
            logger.error(f"Error while splitting the document: {e}")
            return (
                [],
                None,
            )  # Return empty embeddings and None chunks if splitting fails

        embeddings = []
        for chunk in text_chunks:
            try:
                embedding = self.get_embedding(chunk)
                if embedding:
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Empty embedding for chunk: {chunk[:30]}...")
            except Exception as e:
                logger.error(
                    f"Error while generating embedding for chunk: {chunk[:30]}... - {e}"
                )

        return embeddings, text_chunks

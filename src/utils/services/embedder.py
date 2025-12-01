from openai import OpenAI
from src.utils.config import NVIDIA_BASE_URL, NVIDIA_API_KEY, EMBED_MODEL
from src.utils.services.logger_config import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EmbeddingHandler:
    def __init__(self):
        logger.info("Initializing EmbeddingHandler...")

    def get_embedding(self, text: str):

        client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)

        response = client.embeddings.create(
            input=[text],
            model=EMBED_MODEL,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"},
        )

        embedding = response.data[0].embedding
        logger.info(f"Successfully retrieved embedding of length {len(embedding)}")
        return embedding

    def get_document_embeddings(
        self, chunk_size: int, chunk_overlap: int, long_text: str
    ):
        """Splits text into chunks and retrieves embeddings for each chunk."""
        logger.info(f"Splitting text into chunks with size {chunk_size} and overlap {chunk_overlap}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ".", "!", "?", " "],
        )
        text_chunks = splitter.split_text(long_text)

        logger.info("Generated %d text chunks", len(text_chunks))
        embeddings = []

        for i, chunk in enumerate(text_chunks):
            embedding = self.get_embedding(chunk)
            if embedding:
                embeddings.append(embedding)
                logger.info(f"Successfully retrieved embedding for chunk {i + 1}")

        logger.info("Completed generating embeddings for document.")
        return embeddings, text_chunks

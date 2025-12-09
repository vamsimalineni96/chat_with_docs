import requests
from typing import List, Dict, Any
from src.utils.config import RERANK_MODEL, NVIDIA_API_KEY, NVIDIA_RERANK_URL
from src.utils.errors import RerankError
from src.utils.services.logger_config import logger
 
 
class NVidiaReranker:
    def __init__(self):
        self.model = RERANK_MODEL
        self.invoke_url = NVIDIA_RERANK_URL
        self.headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
        }
        self.session = requests.Session()
 
    def execute(
        self, question: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reranks chunks using NVIDIA Reranker while preserving metadata.
        """
        if not question or not isinstance(question, str):
            raise RerankError("Query must be a non-empty string.")
 
        if not retrieved_chunks:
            raise RerankError("Retrieved chunks list is empty — cannot rerank.")
 
        try:
            final_query = {"text": question}
 
            passages = [
                {"text": item["text"], "score": item["score"], "source": item["source"]}
                for item in retrieved_chunks
            ]
 
            rerank_passages = [{"text": p["text"]} for p in passages]
 
            payload = {
                "model": self.model,
                "query": final_query,
                "passages": rerank_passages,
            }
            logger.info("Reranking the retrieved chunks")
            response = self.session.post(
                self.invoke_url, headers=self.headers, json=payload, timeout=20
            )
 
        except requests.exceptions.Timeout:
            raise RerankError("NVIDIA rerank request timed out.")
        except requests.exceptions.RequestException as e:
            raise RerankError(f"Network error calling NVIDIA rerank API: {str(e)}")
 
        # Handle HTTP errors
        if not response.ok:
            try:
                err_msg = response.json().get("message", response.text)
            except Exception:
                err_msg = response.text
            raise RerankError(
                f"NVIDIA rerank API returned {response.status_code}: {err_msg}"
            )
 
        # Parse JSON
        try:
            response_body = response.json()
        except Exception:
            raise RerankError("Failed to decode JSON from rerank API response.")
 
        rankings = response_body.get("rankings")
 
        # Validate rankings
        if rankings is None:
            raise RerankError("Rerank API response missing 'rankings' field.")
 
        if not isinstance(rankings, list) or len(rankings) == 0:
            raise RerankError("Rerank API returned an empty or invalid ranking list.")
 
        try:
            reranked_chunks = [passages[r["index"]] for r in rankings]
        except Exception as e:
            raise RerankError(
                f"Ranking index mismatch between API response and retrieved chunks: {str(e)}"
            )
 
        return reranked_chunks
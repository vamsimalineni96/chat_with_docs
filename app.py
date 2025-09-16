from dotenv import load_dotenv

load_dotenv()

import os
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("files:", os.listdir())

from src.utils.logger_config import LoggerConfig
from src.utils.models import RagCase

from src.rag_flow import RagFlow


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = LoggerConfig().logger


@app.post("/chat")
async def chat(lead: RagCase):
    rag_pipeline = RagFlow()
    
    try:
        logger.info("Generating the response")

        try:
            rag_pipeline.state["user_query"] = lead.question
        except Exception as e:
            raise Exception(f"Failed to set pipeline state: {str(e)}")

        # Step 3: Kick off the pipeline
        try:
            result = await rag_pipeline.kickoff_async()
        except Exception as e:
            raise Exception(f"Pipeline execution failed: {str(e)}")

        return {"user_query": lead.question, "reply": result}

    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in chatbot_evidence: {str(e)}\n{error_trace}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

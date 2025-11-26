from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.rag_pipeline import answer_question
from utils.milvus_store import MilvusStoreHandler
from utils.pdf_parser import PDFParser

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(querry:str, collection_name: str):
    answer = answer_question(question=querry, collection_name= collection_name)
    return {"Answer": answer}


@app.post("/upload_pdf")
async def upload_pdf(pdf_name: str, collection_name: str):
    vector_store = MilvusStoreHandler(collection_name=collection_name)
    pdf_path = f"pdfs/{pdf_name}"

    parser = PDFParser(pdf_path)
    pages = parser.parse_pdf()
    print(f"Total pages parsed: {len(pages)}")

    # Continuous full text of the book
    for i in range(1, len(pages)):
        long_text = pages[i].get("text")
        vector_store.store_in_milvus(text=long_text)
        print(f"Uploaded page: {i} to the vectordb")

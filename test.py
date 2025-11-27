# from pymilvus import MilvusClient

# client = MilvusClient(uri="http://localhost:19530")
# print(client.list_collections())

# if len(client.list_collections())!=0:
#     info = client.describe_collection("rag_nim_milvus")
#     print(info)
#     client.load_collection(collection_name="rag_nim_milvus")
#     rows = client.query(
#         collection_name="rag_nim_milvus",
#         filter="",                      # no filter -> everything
#         output_fields=["id", "doc_id", "source", "chunk_order", "text"],
#         limit=5,                        # just peek at first 5
#     )

#     for r in rows:
#         print("-" * 80)
#         print("id:        ", r.get("id"))
#         print("doc_id:    ", r.get("doc_id"))
#         print("source:    ", r.get("source"))
#         print("order:     ", r.get("chunk_order"))
#         print("text[0:200]:")
#         print((r.get("text") or "")[:200], "...")
# else:
#     print("collection is empty")
#####################################################################
#####################################################################
#####################################################################

from src.utils.services.milvus_store import MilvusStoreHandler
from src.utils.services.pdf_parser import PDFParser

pdf_path = "pdfs\hp.pdf"
parser = PDFParser(pdf_path)
pages = parser.parse_pdf()
print(f"Total pages parsed: {len(pages)}")


# Continuous full text of the book
for i in range(1, len(pages)):
    long_text = pages[i].get("text")
    MilvusStoreHandler().store_in_milvus(text=long_text)
    print(f"Uploaded page: {i} to the vectordb")

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.rankers import TransformersSimilarityRanker

query = "What are effective strategies to improve English speaking skills?"
documents = [
    Document(
        content="Practicing with native speakers enhances English speaking proficiency."
    ),
    Document(
        content="Daily vocabulary expansion is crucial for improving oral communication skills."
    ),
    Document(
        content="Engaging in language exchange programs can significantly boost speaking abilities."
    ),
    Document(
        content="Regular participation in debates and discussions refine public speaking skills in English."
    ),
    Document(
        content="Studying the history of the English language does not directly improve speaking skills."
    ),
]
document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

bm25_retriever = InMemoryBM25Retriever(document_store=document_store)
bm25_docs = bm25_retriever.run(query=query, top_k=4)["documents"]
print("bm25:")
for doc in bm25_docs:
    print(f"content: {doc.content}")
    print(f"score: {doc.score}\n")


reranker = TransformersSimilarityRanker(model="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker.warm_up()
reranked_docs = reranker.run(query=query, documents=bm25_docs, top_k=3)["documents"]
print("reranker:")
for doc in reranked_docs:
    print(f"content: {doc.content}")
    print(f"score: {doc.score}\n")

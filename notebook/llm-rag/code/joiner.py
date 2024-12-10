from haystack import Document, Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersTextEmbedder,
    SentenceTransformersDocumentEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.joiners.document_joiner import DocumentJoiner

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")

query = "What are effective strategies to improve English speaking skills?"
documents = [
    Document(
        content="Practicing with native speakers enhances English speaking proficiency."
    ),
    Document(
        content="Regular participation in debates and discussions refine public speaking skills in English."
    ),
    Document(
        content="Studying the history of the English language does not directly improve speaking skills."
    ),
]

document_embedder = SentenceTransformersDocumentEmbedder(
    model="sentence-transformers/all-MiniLM-L6-v2"
)
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)["documents"]
document_store.write_documents(documents_with_embeddings)

bm25_retriever = InMemoryBM25Retriever(document_store=document_store, scale_score=True)
bm25_docs = bm25_retriever.run(query=query)["documents"]
print("bm25:")
for doc in bm25_docs:
    print(f"content: {doc.content}")
    print(f"score: {doc.score}\n")

query_pipeline = Pipeline()
query_pipeline.add_component(
    "text_embedder",
    SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
)
query_pipeline.add_component(
    "dense_retriever", InMemoryEmbeddingRetriever(document_store=document_store, scale_score=True)
)
query_pipeline.connect("text_embedder.embedding", "dense_retriever.query_embedding")
dense_docs = query_pipeline.run({"text_embedder": {"text": query}})["dense_retriever"][
    "documents"
]
print("dense:")
for doc in dense_docs:
    print(f"content: {doc.content}")
    print(f"score: {doc.score}\n")

joiner = DocumentJoiner(join_mode="merge", weights=[0.3, 0.7])
merge_docs = joiner.run(documents=[bm25_docs, dense_docs])["documents"]
print("merge:")
for doc in merge_docs:
    print(f"content: {doc.content}")
    print(f"score: {doc.score}\n")

joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")
rrf_docs = joiner.run(documents=[bm25_docs, dense_docs])["documents"]
print("rrf:")
for doc in rrf_docs:
    print(f"content: {doc.content}")
    print(f"score: {doc.score}\n")
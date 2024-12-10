from haystack import Document, Pipeline
from haystack_integrations.components.retrievers.qdrant import (
    QdrantSparseEmbeddingRetriever,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)


document_store = QdrantDocumentStore(
    ":memory:", recreate_index=True, use_sparse_embeddings=True, similarity="cosine"
)

documents = [
    Document(content="My name is Wolfgang and I live in Berlin"),
    Document(content="I saw a black horse running"),
    Document(content="Germany has many big cities"),
    Document(content="fastembed is supported by and maintained by Qdrant."),
]

sparse_document_embedder = FastembedSparseDocumentEmbedder(
    model="prithivida/Splade_PP_en_v1"
)
sparse_document_embedder.warm_up()
documents_with_embeddings = sparse_document_embedder.run(documents)["documents"]
document_store.write_documents(documents_with_embeddings)
for doc in documents_with_embeddings:
    print(f"content: {doc.content}")
    print(f"score: {doc.score}")
    print(f"sparse_embedding: {doc.sparse_embedding}\n")

query_pipeline = Pipeline()
query_pipeline.add_component(
    "sparse_text_embedder",
    FastembedSparseTextEmbedder(model="prithivida/Splade_PP_en_v1"),
)
query_pipeline.add_component(
    "sparse_retriever",
    QdrantSparseEmbeddingRetriever(document_store=document_store),
)
query_pipeline.connect(
    "sparse_text_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding"
)

query = "Who supports fastembed?"
result = query_pipeline.run({"sparse_text_embedder": {"text": query}})
result_documents = result["sparse_retriever"]["documents"]
for doc in result_documents:
    print(f"content: {doc.content}")
    print(f"score: {doc.score}\n")

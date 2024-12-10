from rerankers import Reranker

ranker = Reranker("answerdotai/answerai-colbert-small-v1", model_type="colbert")
docs = [
    "There are over 7,000 languages spoken around the world today.",
    "Elephants have been observed to behave in a way that indicates a high level of self-awareness, such as recognizing themselves in mirrors.",
    "In certain parts of the world, like the Maldives, Puerto Rico, and San Diego, you can witness the phenomenon of bioluminescent waves.",
]
query = "How many languages are there?"
reranked_results = ranker.rank(query=query, docs=docs)
for result in reranked_results.results:
    print(f"content: {result.document.text}")
    print(f"score: {result.score}\n")

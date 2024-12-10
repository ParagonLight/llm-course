from haystack.components.converters import TextFileToDocument

converter = TextFileToDocument()
docs = converter.run(sources=["./files/hello_world.txt"])["documents"]

print(f"id: {docs[0].id}")
print(f"content: {docs[0].content}")
print(f"score: {docs[0].score}")
print(f"embedding: {docs[0].embedding}")
print(f"sparse_embedding: {docs[0].sparse_embedding}")
print(f"meta: {docs[0].meta}")

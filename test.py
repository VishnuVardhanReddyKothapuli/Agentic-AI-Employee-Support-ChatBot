from itertools import count
from app.services.ingestion import load_file, chunk_documents
from pathlib import Path

docs = load_file('data\hr-data\HR-Emp Policies.pdf')
chunked_docs = chunk_documents(docs)
print(chunked_docs)

print(len(chunked_docs))
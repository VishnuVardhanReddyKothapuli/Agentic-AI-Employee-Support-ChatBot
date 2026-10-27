from pathlib import Path
from app.services.ingestion import load_file, chunk_documents

pdf_path = Path("data") / "hr-data" / "HR-Emp Policies.pdf"
if pdf_path.exists():
    docs = load_file(pdf_path)
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} pages, created {len(chunks)} chunks.")
    if chunks:
        print("Sample chunk metadata:", chunks[0].metadata)
else:
    print(f"File not found: {pdf_path}")

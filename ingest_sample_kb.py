from pathlib import Path
from app.core.config import get_settings
from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents


def ingest_knowledge_base():
    settings = get_settings()
    folder = Path(settings.sample_kb_dir)
    if not folder.exists() or not any(folder.iterdir()):
        alt_folder = Path("data/hr-data")
        if alt_folder.exists():
            folder = alt_folder

    print(f"Loading documents from: {folder.resolve()}")
    files = [p for p in folder.iterdir() if p.is_file()]
    if not files:
        print("No files found to ingest.")
        return

    all_docs = []
    for path in files:
        print(f"  - Loading {path.name}")
        all_docs.extend(load_file(path))

    print(f"Loaded {len(all_docs)} raw document pages/sections.")
    chunks = chunk_documents(all_docs)
    print(f"Generated {len(chunks)} text chunks.")

    print("Uploading to Pinecone...")
    ids = add_documents(chunks)
    print(f"Successfully indexed {len(ids)} document chunks in Pinecone.")


if __name__ == "__main__":
    ingest_knowledge_base()

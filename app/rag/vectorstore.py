from pinecone.inference.models.embedding_list import OpenAPIEmbeddingsList
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()
    if not name:
        raise RuntimeError("Embedding model is not configured")
    normalized = name.lower()
    if normalized in EMBEDDING_DIMENSIONS:
        return EMBEDDING_DIMENSIONS[normalized]
    if "text-embedding-3-small" in normalized:
        return 1536
    if "text-embedding-3-large" in normalized:
        return 3072
    if "text-embedding-ada-002" in normalized:
        return 1536
    if "all-minilm" in normalized:
        return 384
    raise ValueError(
        f"Unsupported embedding model '{model_name or settings.embedding_model}' for Pinecone. "
        "Add the matching dimension to EMBEDDING_DIMENSIONS."
    )



def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        _embeddings = HuggingFaceEmbeddings(
            model=settings.embedding_model,
            api_key=settings.groq_api_key,
        )
    return _embeddings



def ensure_index():
    import os

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")
    desired_dimension = get_embedding_dimension()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    index_name = (
        getattr(settings, "pinecone_index_name", None)
        or os.getenv("PINECONE_INDEX_NAME")
    )
    if not index_name:
        existing_indexes = pc.list_indexes().names()
        index_name = existing_indexes[0] if existing_indexes else "agentic-hr-emp-bot"

    if pc.has_index(index_name):
        index_info = pc.describe_index(index_name)
        current_dimension = getattr(index_info, "dimension", None)
        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")
        if current_dimension is not None and current_dimension != desired_dimension:
            pc.delete_index(name=index_name)
            while pc.has_index(index_name):
                time.sleep(1)

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while True:
            status = getattr(pc.describe_index(index_name), "status", None)
            is_ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
            if is_ready:
                break
            time.sleep(1)

    return pc.Index(index_name)



def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore



def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})




def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)
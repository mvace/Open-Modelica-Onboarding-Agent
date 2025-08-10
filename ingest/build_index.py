# ingest/build_index.py
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from settings import settings
from ingest.loaders import load_and_split_pdf


def build_faiss_index():
    print("Loading and splitting PDF...")
    docs = load_and_split_pdf()
    print(f"Loaded {len(docs)} chunks.")

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs(settings.index_dir, exist_ok=True)
    vectorstore.save_local(settings.index_dir)
    print(f"FAISS index saved to {settings.index_dir}")


if __name__ == "__main__":
    build_faiss_index()

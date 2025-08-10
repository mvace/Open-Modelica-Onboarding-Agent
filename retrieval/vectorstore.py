from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from settings import settings


def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    return FAISS.load_local(
        settings.index_dir, embeddings, allow_dangerous_deserialization=True
    )


def get_retriever(k: int = 4):
    return get_vectorstore().as_retriever(search_kwargs={"k": k})

from retrieval.vectorstore import get_retriever

if __name__ == "__main__":
    retriever = get_retriever(k=3)
    query = "How do I install OpenModelica?"
    results = retriever.get_relevant_documents(query)
    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i} (page {doc.metadata.get('page')}):\n")
        print(doc.page_content[:300], "...")

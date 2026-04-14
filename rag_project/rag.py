import hashlib
import diskcache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "./chroma_db"
CACHE_DIR = "./cache"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "tinyllama"

cache = diskcache.Cache(CACHE_DIR, size_limit=2**30)

def make_cache_key(query):
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()

print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"}
)

print("Loading vector store...")
vectordb = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

print("Connecting to Ollama...")
llm = Ollama(model=LLM_MODEL, temperature=0.1)

PROMPT = PromptTemplate(
    template="Use the context below to answer the question.\nIf the answer is not in the context, say I dont know.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

def query(question):
    key = make_cache_key(question)
    if key in cache:
        print("[CACHE HIT] Returning cached answer")
        return cache[key]
    print("[CACHE MISS] Asking LLM...")
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    response = {
        "answer": answer,
        "sources": [doc.metadata.get("source", "unknown") for doc in docs]
    }
    cache.set(key, response, expire=3600)
    return response

if __name__ == "__main__":
    print("RAG ready! Type your questions.")
    while True:
        q = input("\nYour question (or quit): ")
        if q.lower() == "quit":
            break
        out = query(q)
        print(f"\nAnswer: {out['answer']}")
        print(f"Sources: {out['sources']}")
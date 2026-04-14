from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "./data"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def ingest():
    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Embedding... first run downloads 90MB model")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print("Done! Vector store saved.")

if __name__ == "__main__":
    ingest()
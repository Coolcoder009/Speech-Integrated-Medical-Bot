from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
Path = "Data/"

def load_file(data):
    loader = DirectoryLoader(data, glob = '*.pdf', loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_file(data = Path)


def create_chunks(extract_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extract_data)
    return text_chunks

text_chunks = create_chunks(extract_data = documents)


def get_embeddings():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embeddings()


DB_Path = "vectorstores/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_Path)
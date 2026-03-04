from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from rag.core.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
load_dotenv()


if settings.VECTOR_DATABASE_PATH.exists():
    print("Vector store already exists. Skipping ingestion.")
    exit()


# ---------------------
# Loading the documents
# ---------------------
loader = DirectoryLoader(
    path = str(settings.DATA_PATH),
    glob="**/*.pdf", 
    loader_cls = UnstructuredPDFLoader
)
docs = loader.load()                        # can use lazy loading for huge number of documents
print("Number of docs : ", len(docs))



# ---------
# Chunking 
# ---------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 60,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunked_text = text_splitter.split_documents(documents=docs)
print("Number of chunks : ", len(chunked_text))



# --------------------------------
# Initializing the vector database 
# --------------------------------
# embedder = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
embedder = HuggingFaceEmbeddings(model_name=settings.HF_EMBEDDER)
vector_database = Chroma.from_documents(
    documents=chunked_text,
    embedding=embedder,
    persist_directory=str(settings.VECTOR_DATABASE_PATH)
)
vector_database.persist()
print("Vector store initialized successfully")
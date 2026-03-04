from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever 
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from rag.core.config import settings
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
load_dotenv()

embedder = HuggingFaceEmbeddings(model_name=settings.HF_EMBEDDER)
llm = HuggingFaceEndpoint(repo_id=settings.HF_LLM)
model = ChatHuggingFace(llm=llm)

def get_retriever():
    """
    Returns a MultiQueryRetriever built on ChromaDB
    """

    # loading the persisted vectorstore
    vectorstore = Chroma(
        persist_directory=str(settings.VECTOR_DATABASE_PATH),
        embedding_function=embedder
    )

    # Initializing the base similarity retriever
    base_retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = {"k" : 5}
    )

    # Multiquery retriever
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm = model
    )

    return mq_retriever
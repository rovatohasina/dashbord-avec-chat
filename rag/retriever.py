from embedding.doc_embeddings import embed_pdfs_and_wbdata_and_check, get_pdf_documents
from data.wbdata_loader import get_live_wbdata
from rag.pinecone import init_pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
def create_retriever():
    region = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_client, _ = init_pinecone()
    index_name = "nouvel"
    
    index_pinecone = pinecone_client.Index(index_name)
    
    documents = get_pdf_documents("data/doc")
    df = get_live_wbdata()

    pdf_vectors, wbdata_vectors = embed_pdfs_and_wbdata_and_check(df, documents)

    # Diviser les documents en chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_docs = splitter.split_documents(documents)
    
    # Initialisation du modèle d'Embed HuggingFace
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/multilingual-Mini-LM-L12-v2")
    pdf_vector_store = LangchainPinecone.from_documents(splitted_docs, embed_model, index_name=index_name)
    retriever = pdf_vector_store.as_retriever()
    return retriever










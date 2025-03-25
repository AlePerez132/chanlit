import os
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

dir="./pdf"

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
loader = DirectoryLoader(
    dir,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)

docs = loader.load()

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
split_docs = splitter.split_documents(docs)

# Extraer solo el texto de los documentos
texts = [doc.page_content for doc in split_docs]

# Create embeddings using OpenAI
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

from langchain_community.vectorstores import FAISS
db = FAISS.from_texts(texts, embeddings)
db.save_local("faiss_index")
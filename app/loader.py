from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
import os

def load_documents_from_folder(folder_path: str) -> list[Document]:
    documents = []
    folder_path = "data"
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename))
            documents.extend(loader.load())
        elif filename.endswith(".json"):
            loader = JSONLoader(folder_path)
            documents.extend(loader.load())
        elif filename.endswith(".csv"):
            loader = CSVLoader(folder_path)
            documents.extend(loader.load())
    return documents

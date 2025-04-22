from app.embeddings import load_embedding_model
from app.loader import load_documents_from_folder
from app.milvus import init_milvus
from app.indexer import index_documents
from app.search import search_query

# Load embedding model
embedding = load_embedding_model()

# Initialize Milvus
vectorstore = init_milvus("e5_collection", embedding)

# Load documents from folder and index them
documents = load_documents_from_folder("data")
index_documents(vectorstore, documents)

# Test a query
query = "What's the purpose of Milvus?"
results = search_query(vectorstore, query)

print("\n🔍 Search Results:")
for res in results:
    print("-", res)

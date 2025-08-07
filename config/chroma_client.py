import chromadb
from chromadb.utils import embedding_functions

def init_chroma():
    """Initialize and return ChromaDB collection"""
    client = chromadb.PersistentClient(path="chroma_db")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="hr_docs",
        embedding_function=embedding_func
    )


#Consistent Embedding Initialization

# def init_chroma():
#     # Define your embedding function FIRST
#     embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name="all-MiniLM-L6-v2"
#     )
    
#     # Initialize client with persistence
#     client = chromadb.PersistentClient(path="hr_chroma_db")
    
#     # Get or create collection with the SAME embedder
#     return client.get_or_create_collection(
#         name="hr_docs",
#         embedding_function=embedder  # Consistent embedding function

#     )

#Force Recreate Collection (For Development)

# def init_chroma():
#     client = chromadb.PersistentClient(path="chroma_db")
    
#     # Delete old collection if exists
#     try:
#         client.delete_collection("hr_docs")
#     except:
#         pass
        
#     # Create new with your embedder
#     return client.create_collection(
#         name="hr_docs",
#         embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name="all-MiniLM-L6-v2"
#         )
#     )
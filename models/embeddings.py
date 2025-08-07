from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

class HREmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_collection("hr_docs")
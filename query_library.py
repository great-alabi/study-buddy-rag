import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Pinecone and local AI translation model
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "research-papers-index"
index = pc.Index(index_name)

print("Loading local AI model for searching...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_library(query_text, top_k=3):
    print(f"\nSearching for: '{query_text}'\n" + "-"*50)
    
    # Translate the user's question into meaning numbers
    query_vector = model.encode(query_text).tolist()
    
    # Query Pinecone for the closest matching text chunks
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
        timeout = 15
    )
    
    for i, match in enumerate(results.matches, 1):
        score = match.score
        source = match.metadata.get("source", "Unknown")
        snippet = match.metadata.get("text", "")
        
        print(f"Match {i} (Confidence Score: {score:.4f})")
        print(f"Source Paper: {source}")
        print(f"Snippet:\n{snippet}\n")
        print("-" * 50)

if __name__ == "__main__":
   

    search_library(user_query)

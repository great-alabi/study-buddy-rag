import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("\n[Error]: GEMINI_API_KEY not found in environment variables!")
    sys.exit(1)

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("research-papers-index")
model = SentenceTransformer('all-MiniLM-L6-v2')
ai_client = genai.Client(api_key=api_key)

def get_indexed_sources():
    try:
        dummy_vector = [0.0] * 384
        results = index.query(vector=dummy_vector, top_k=100, include_metadata=True)
        sources = set()
        for match in results.get('matches', []):
            source = match.get('metadata', {}).get('source_paper')
            if source:
                sources.add(source)
        return list(sources)
    except Exception:
        return []

def query_rag(user_query):
    print(f"\nQuerying library for: '{user_query}'\n")
    
    all_sources = get_indexed_sources()
    inventory_str = ", ".join(all_sources) if all_sources else "Unknown"
    
    query_vector = model.encode(user_query).tolist()
    
    # Retrieve top chunks for semantic context
    search_results = index.query(
        vector=query_vector,
        top_k=6,
        include_metadata=True
    )
    
    context_chunks = []
    for match in search_results.get('matches', []):
        source = match['metadata'].get('source_paper', 'Unknown')
        text = match['metadata'].get('text', '')
        context_chunks.append(f"Source: {source}\nSnippet: {text}")
        
    combined_context = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""
    You are an expert research assistant managing a local library of documents. 
    
    IMPORTANT SYSTEM INSTRUCTION FOR COUNTS/INVENTORY:
    - If the user asks about how many files/papers are in the library, what files are indexed, or requests a file inventory, you MUST use the Library Inventory list below. Do NOT look inside the text chunks for bibliometric stats or document counts.

    Library Inventory:
    - Total files: {len(all_sources)}
    - File names/sources: {inventory_str}

    Extracted Content Chunks:
    {combined_context}

    User Question: {user_query}

    Instructions: Answer the user's question accurately using the Library Inventory and the extracted context chunks.
    """
    
    max_retries = 3
    delay = 3
    
    for attempt in range(max_retries):
        try:
            response = ai_client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt
            )
            print("--- AI Generated Answer ---")
            print(response.text)
            return
        except Exception as e:
            print(f"[Attempt {attempt + 1}/{max_retries}] API connection hiccup: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                print("\n[API Error]: Maximum retries reached.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        query_rag(user_input)
    else:
        print("Please provide a query.")
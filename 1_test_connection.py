import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "research-papers-index"

print("--- Checking Pinecone Connection ---")
if index_name in [index.name for index in pc.list_indexes()]:
    print(f"Success! Found index: '{index_name}'")
else:
    print(f"Index '{index_name}' not found. Please create it first.")

print("\n--- Checking Research Papers Directory ---")
pdf_dir = "research_papers"
if os.path.exists(pdf_dir):
    files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDF(s) in '{pdf_dir}/':")
    for f in files:
        print(f" - {f}")
else:
    print(f"Directory '{pdf_dir}' does not exist.")

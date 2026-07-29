import os
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize Pinecone and local AI translation model
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "research-papers-index"
index = pc.Index(index_name)

print("Loading local AI embedding model (this may take a moment on first run)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

PAPERS_DIR = Path("research_papers")
CHUNK_SIZE = 1000  # 1000 characters per bite-sized piece

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Splits text into bite-sized chunks of roughly chunk_size characters."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_and_upload():
    if not PAPERS_DIR.exists():
        print(f"Directory '{PAPERS_DIR}' not found!")
        return

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process.\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        try:
            reader = PdfReader(pdf_path)
            full_text = ""
            for page_idx, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    full_text += extracted + "\n"

            # Chop into bite-sized paragraphs
            chunks = chunk_text(full_text)
            print(f"  - Total characters: {len(full_text)} | Total chunks created: {len(chunks)}")

            # Generate local embeddings and prepare vectors for Pinecone
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                # Translate text chunk into meaning numbers locally
                vector = model.encode(chunk).tolist()
                vector_id = f"{pdf_path.stem}-chunk-{i}"
                
                vectors_to_upsert.append({
                    "id": vector_id,
                    "values": vector,
                    "metadata": {"source": pdf_path.name, "text": chunk[:500]} # Store text snippet
                })

            # Upload to Pinecone in batches of 100
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                index.upsert(vectors=batch)
            
            print(f"  - Successfully uploaded {len(vectors_to_upsert)} chunks to Pinecone!\n")

        except Exception as e:
            print(f"  - Error processing {pdf_path.name}: {e}")

if __name__ == "__main__":
    process_and_upload()

print("1. Script started successfully!")

import os
from dotenv import load_dotenv
from pinecone import Pinecone

print("2. Imports successful!")

# Load environment variables (reads your .env file for PINECONE_API_KEY)
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    print("Error: PINECONE_API_KEY not found in environment variables.")
else:
    print("3. Pinecone API key loaded successfully!")
    # Initialize Pinecone client (using 384 dimensions as per your setup)
    pc = Pinecone(api_key=api_key)
    print("4. Successfully connected to Pinecone client!")

    # Optional: List your indexes to confirm connectivity
    indexes = pc.list_indexes().names()
    print(f"Available Pinecone Indexes: {list(indexes)}")
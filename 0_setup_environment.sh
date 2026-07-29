#!/bin/bash
set -e
echo "=== Installing project dependencies ==="
/c/Python314/python.exe -m pip install --upgrade pip
/c/Python314/python.exe -m pip install pinecone sentence-transformers pypdf google-genai python-dotenv
echo "=== Setup Complete! ==="


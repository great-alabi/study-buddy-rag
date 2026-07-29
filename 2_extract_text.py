import os
from pypdf import PdfReader

pdf_dir = "research_papers"

print("--- Extracting Text from PDFs ---")
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        path = os.path.join(pdf_dir, filename)
        reader = PdfReader(path)
        total_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                total_text += text
        print(f"Processed: {filename} ({len(total_text)} characters extracted)")

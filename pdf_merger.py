import os
from pypdf import PdfReader, PdfWriter

# ======= USER SETTINGS =======
pdf_folder = "/Users/bethlarsen/Downloads/pdfs/temp_pdf"
output_file = "/Users/bethlarsen/Downloads/pdfs/combined/midterm1.pdf"
# ==============================

writer = PdfWriter()

# Get all PDFs and sort alphabetically
pdf_files = sorted(
    [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
)

if not pdf_files:
    raise ValueError("No PDF files found in the folder.")

print(f"Found {len(pdf_files)} PDFs. Merging...")

for pdf_name in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_name)
    print(f"Adding {pdf_name}")

    reader = PdfReader(pdf_path)
    for page in reader.pages:
        writer.add_page(page)

with open(output_file, "wb") as f:
    writer.write(f)

print(f"\nDone! Combined PDF saved to:\n{output_file}")
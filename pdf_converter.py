import os
import subprocess
import pypdf
from pathlib import Path
from pypdf import PdfWriter

# ===== USER SETTINGS =====
input_folder = "/Users/bethlarsen/Downloads/pdfs"
temp_pdf_folder = "/Users/bethlarsen/Downloads/pdfs/temp_pdf"

os.makedirs(temp_pdf_folder, exist_ok=True)


def convert_ppt_to_pdf(ppt_path, pdf_path):
    """
    Uses AppleScript to tell PowerPoint to export a PPT as PDF
    """
    script = f'''
    tell application "Microsoft PowerPoint"
        open POSIX file "{ppt_path}"
        save active presentation in POSIX file "{pdf_path}" as save as PDF
        close active presentation
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=True)


# ===== Convert all PPTX files =====
pdf_files = []

for file in sorted(os.listdir(input_folder)):
    if file.lower().endswith(".pptx"):
        ppt_path = os.path.join(input_folder, file)
        pdf_path = os.path.join(temp_pdf_folder, file.replace(".pptx", ".pdf"))

        print(f"Converting {file}...")
        convert_ppt_to_pdf(ppt_path, pdf_path)
        pdf_files.append(pdf_path)

# ===== Merge PDFs =====


folder = "/Users/bethlarsen/Downloads/pdfs/combined"
output = "combined.pdf"

writer = PdfWriter()

for file in sorted(os.listdir(folder)):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(folder, file)
        print(f"Adding {file}")

        from pypdf import PdfReader

        reader = PdfReader(pdf_path)

        for page in reader.pages:
            writer.add_page(page)

with open(output, "wb") as f:
    writer.write(f)

print("Done!")


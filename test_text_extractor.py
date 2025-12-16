from modules.text_extractor import extract_text
from pathlib import Path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Define test files
test_files = [
    ("sample_files/sample.pdf", "pdf"),
    # ("sample_files/sample.docx", "docx"),
    # ("sample_files/sample.pptx", "pptx"),
    # ("sample_files/handwritten.jpg", "jpg"),
    # ("sample_files/handwritten.png", "png"),
]

for file_path, file_type in test_files:
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️ File not found: {file_path}")
        continue

    print(f"\n--- Testing {file_type.upper()} Extraction ---")
    try:
        text = extract_text(str(path), file_type)
        print(f"✅ Successfully extracted {len(text)} characters.")
        print(f"🔹 Sample Output:\n{text[:300]}...\n")
    except Exception as e:
        print(f"❌ Error processing {file_type}: {e}")

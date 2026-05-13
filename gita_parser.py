import fitz  # PyMuPDF
import json
import re
import os

def parse_gita_pdf(pdf_path, output_json_path):
    # Verify the file actually exists before trying to open it
    if not os.path.exists(pdf_path):
        print(f"ERROR: Could not find the file at {pdf_path}")
        print("Make sure you are running this script from the correct folder!")
        return

    print(f"Opening {pdf_path}...")
    doc = fitz.open(pdf_path)
    
    parsed_data = []
    
    # State tracking variables
    current_chapter = "Unknown"
    current_text_num = None
    
    current_sanskrit = []
    current_translation = []
    current_purport = []
    
    # States: "WAITING", "SANSKRIT", "TRANSLATION", "PURPORT"
    current_state = "WAITING"

    # Regex patterns
    chapter_pattern = re.compile(r"^CHAPTER\s+([A-Z]+|\d+)", re.IGNORECASE)
    text_pattern = re.compile(r"^TEXTS?\s+(\d+.*)", re.IGNORECASE)

    print("Parsing pages... This might take a few seconds.")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Ignore obvious page headers/footers
            if "Bhagavad-gita As It Is" in line or line.isdigit():
                continue

            # 1. Check for Chapter
            chapter_match = chapter_pattern.match(line)
            if chapter_match:
                current_chapter = chapter_match.group(1)
                continue

            # 2. Check for New TEXT (Verse)
            text_match = text_pattern.match(line)
            if text_match:
                # Save the PREVIOUS verse before starting a new one
                if current_text_num:
                    parsed_data.append({
                        "chapter": current_chapter,
                        "verse": current_text_num,
                        "sanskrit": " ".join(current_sanskrit).strip(),
                        "translation": " ".join(current_translation).strip(),
                        "purport": " ".join(current_purport).strip()
                    })
                
                # Reset for the new verse
                current_text_num = text_match.group(1)
                current_sanskrit, current_translation, current_purport = [], [], []
                current_state = "SANSKRIT"  
                continue

            # 3. Check for TRANSLATION trigger
            if line == "TRANSLATION":
                current_state = "TRANSLATION"
                continue
                
            # 4. Check for PURPORT trigger
            if line == "PURPORT":
                current_state = "PURPORT"
                continue

            # 5. Append text based on the current state
            if current_state == "SANSKRIT":
                current_sanskrit.append(line)
            elif current_state == "TRANSLATION":
                current_translation.append(line)
            elif current_state == "PURPORT":
                current_purport.append(line)

    # Don't forget to save the very last verse in the book!
    if current_text_num:
        parsed_data.append({
            "chapter": current_chapter,
            "verse": current_text_num,
            "sanskrit": " ".join(current_sanskrit).strip(),
            "translation": " ".join(current_translation).strip(),
            "purport": " ".join(current_purport).strip()
        })

    # Write to JSON
    print(f"Extraction complete! Found {len(parsed_data)} verses.")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=4, ensure_ascii=False)
    print(f"Saved cleanly to {output_json_path}")


# --- RUN THE SCRIPT ---
# Updated with your exact relative path
PDF_FILE = "../data/pdf/11-Bhagavad-gita_As_It_Is.pdf" 
OUTPUT_FILE = "gita_structured.json"

parse_gita_pdf(PDF_FILE, OUTPUT_FILE)
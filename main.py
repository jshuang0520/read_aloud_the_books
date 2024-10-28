import PyPDF2
import pyttsx3
import argparse

def extract_text_from_pdf(pdf_path, start_page=0):
    print("Opening the PDF file...")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        num_pages = len(reader.pages)
        print(f"Total pages in PDF: {num_pages}")

        if start_page >= num_pages:
            raise ValueError(f"Start page {start_page} exceeds total number of pages {num_pages}.")

        for page_num in range(start_page, num_pages):
            print(f"Extracting text from page {page_num + 1}...")
            text += reader.pages[page_num].extract_text()
    print("Finished extracting text from PDF.")
    return text

def read_aloud(text, rate=150):
    print("Initializing text-to-speech engine...")
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)  # Adjust the speed
    print("Starting to read aloud...")
    engine.say(text)
    engine.runAndWait()
    print("Finished reading aloud.")

def read_pdf_aloud(pdf_path, rate=150, start_page=0):
    text = extract_text_from_pdf(pdf_path, start_page=start_page)
    read_aloud(text, rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read PDF aloud with adjustable speed and start page")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--rate", type=int, default=150, help="Reading speed in words per minute")
    parser.add_argument("--start_page", type=int, default=0, help="Page number to start reading from (0-based index)")
    args = parser.parse_args()

    print(f"Reading from {args.pdf_path} at speed {args.rate} WPM, starting from page {args.start_page + 1}")
    read_pdf_aloud(args.pdf_path, rate=args.rate, start_page=args.start_page)


import fitz  # PyMuPDF
from tqdm import tqdm
import logging
import os
import gc
import re
import string
from num2words import num2words
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
from pydub.effects import normalize
import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import numpy as np
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile as wavfile

nltk.download('punkt', quiet=True)

# Config
file_name = 'Storytelling_with_Data'
pdf_path = f'/Users/johnson.huang/py_ds/read_aloud_the_books/books/{file_name}.pdf'
audio_folder = f'audio_output/{file_name}'
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

# Set up logging
log_folder = 'logs'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_file = os.path.join(log_folder, f'{file_name}.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])
logging.info(f"Starting to process PDF: {pdf_path}")

# # Initialize TTS model
# processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Initialize TTS model
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

# Speaker embedding
speaker_embedding = torch.randn(1, 512)  # You can fine-tune this for different voices


def preprocess_text(text):
    # Normalize abbreviations and symbols
    abbreviations = {
        'Dr.': 'Doctor',
        'Mr.': 'Mister',
        'Mrs.': 'Misses',
        'Prof.': 'Professor',
        'e.g.': 'for example',
        'i.e.': 'that is',
        'etc.': 'et cetera',
        '#': 'number',
        '%': 'percent',
        '&': 'and',
    }
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)

    # Handle hyphenated words
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)  # Join words split across lines
    text = re.sub(r'(\w+)-(\w+)', lambda m: f"{m.group(1)} {m.group(2)}" if m.group(1).lower() in ['three', 'minute', 'axis'] else m.group(0), text)

    # Replace numbers with words
    text = re.sub(r'\b(\d+)\b', lambda m: num2words(int(m.group(0))), text)

    # Handle apostrophes
    text = re.sub(r"(\w+)'s", r"\1 is", text)
    text = re.sub(r"(\w+)n't", r"\1 not", text)
    text = re.sub(r"(\w+)'re", r"\1 are", text)
    text = re.sub(r"(\w+)'ll", r"\1 will", text)
    text = re.sub(r"(\w+)'ve", r"\1 have", text)
    text = re.sub(r"(\w+)'d", r"\1 would", text)

    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.,?!-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Improve sentence segmentation using NLTK
    sentences = sent_tokenize(text)

    # Process each sentence
    processed_chunks = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(0, len(words), 20):  # Increased chunk size
            chunk = ' '.join(words[i:i+20])
            if chunk:
                processed_chunks.append(chunk)

    # Final processing of chunks
    final_chunks = []
    for chunk in processed_chunks:
        chunk = chunk.capitalize()
        chunk = ''.join(filter(lambda x: x in string.printable, chunk))
        if chunk.strip() and not chunk.isspace() and len(chunk.split()) >= 3:
            final_chunks.append(chunk)

    return final_chunks


def generate_audio(text_chunks, audio_path):
    full_audio = np.array([])
    for i, chunk in enumerate(text_chunks):
        try:
            inputs = tokenizer(chunk, return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            audio = output.numpy().flatten()
            full_audio = np.concatenate((full_audio, audio))
        except Exception as e:
            logging.error(f"Error generating audio for chunk {i}: {str(e)}")

    # Normalize audio
    full_audio = full_audio / np.max(np.abs(full_audio))

    # Save as wav file
    wavfile.write(audio_path, rate=16000, data=(full_audio * 32767).astype(np.int16))
    return audio_path


def post_process_audio(audio_path):
    audio = AudioSegment.from_wav(audio_path)

    # Normalize audio
    audio = normalize(audio)

    # Remove silence at the beginning and end
    silence_thresh = -50
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
    if nonsilent_ranges:
        start_trim, end_trim = nonsilent_ranges[0][0], nonsilent_ranges[-1][1]
        audio = audio[start_trim:end_trim]

    # Export processed audio
    audio.export(audio_path, format="wav")


def process_batch(start_page, end_page):
    doc = fitz.open(pdf_path)
    for page_num in range(start_page, end_page):
        page = doc[page_num]
        text = page.get_text()
        # Preprocess the text
        processed_chunks = preprocess_text(text)
        logging.info(f'Processed contents of page {page_num}: \n{processed_chunks}')
        if processed_chunks:
            audio_path = os.path.join(audio_folder, f'page_{page_num:03d}.wav')
            logging.info(f"Generating audio for page {page_num}...")
            try:
                # Generate audio with Hugging Face model
                generate_audio(processed_chunks, audio_path)
                # Post-process the audio
                post_process_audio(audio_path)
                logging.info(f"Audio file created and processed: {audio_path}")
            except Exception as e:
                logging.error(f"Error generating audio for page {page_num}: {str(e)}")
        else:
            logging.warning(f"Page {page_num} is empty or contains no valid text after processing")
        page = None
    doc.close()
    gc.collect()


# Get total number of pages
with fitz.open(pdf_path) as doc:
    total_pages = len(doc)

logging.info(f"Total pages in PDF: {total_pages}")

# Process in batches
batch_size = 1
for batch_start in tqdm(range(0, total_pages, batch_size), desc="Processing batches", unit="batch"):
    batch_end = min(batch_start + batch_size, total_pages)
    process_batch(batch_start, batch_end)

logging.info("Audio generation complete")



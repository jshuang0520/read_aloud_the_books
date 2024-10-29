import fitz  # PyMuPDF
from TTS.api import TTS
from tqdm import tqdm
import logging
import os
import gc
import re
import string
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent


# config
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

# Initialize TTS with enhanced settings
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True)


def preprocess_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s.,?!-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split long text into sentences

    # Process each sentence
    processed_chunks = []
    for sentence in sentences:
        # Split long sentences at punctuation marks or every 25 words
        parts = re.split(r'([,;:])', sentence)
        current_chunk = ""
        word_count = 0
        for part in parts:
            words = part.split()
            for word in words:
                if word_count >= 25 or (current_chunk and current_chunk[-1] in '.!?'):
                    if current_chunk.strip():  # Only add non-empty chunks
                        processed_chunks.append(current_chunk.strip())
                    current_chunk = word
                    word_count = 1
                else:
                    current_chunk += " " + word
                    word_count += 1

            if part in ',;:':
                current_chunk += part
                if word_count >= 20:  # Split a bit earlier if we're near the limit
                    if current_chunk.strip():  # Only add non-empty chunks
                        processed_chunks.append(current_chunk.strip())
                    current_chunk = ""
                    word_count = 0

        if current_chunk.strip():  # Only add non-empty chunks
            processed_chunks.append(current_chunk.strip())

    # Final processing of chunks
    final_chunks = []
    for chunk in processed_chunks:
        # Remove multiple punctuation marks
        chunk = re.sub(r'([.,?!;:])\1+', r'\1', chunk)
        # Remove trailing punctuation except for sentence-ending punctuation
        chunk = re.sub(r'[,;:]$', '', chunk)
        # Capitalize first letter of each chunk
        chunk = chunk.capitalize()
        # Remove any remaining non-printable characters
        chunk = ''.join(filter(lambda x: x in string.printable, chunk))
        # Only add non-empty chunks with at least 5 words
        if chunk.strip() and not chunk.isspace() and len(chunk.split()) >= 5:
            final_chunks.append(chunk)

    return final_chunks


def remove_strange_sounds(audio):
    # Split audio on silence
    chunks = split_on_silence(audio, min_silence_len=100, silence_thresh=-40)

    good_chunks = []
    for chunk in chunks:
        # Check if the chunk has a reasonable duration and volume
        if len(chunk) > 100 and chunk.dBFS > -35:
            # Further split the chunk to check for sudden volume changes
            sub_chunks = split_on_silence(chunk, min_silence_len=50, silence_thresh=-35)
            if all(abs(c.dBFS - chunk.dBFS) < 15 for c in sub_chunks):
                good_chunks.append(chunk)
            else:
                # If there are sudden volume changes, keep the longer sub-chunks
                good_chunks.extend([c for c in sub_chunks if len(c) > 200])

    # Combine good chunks
    return sum(good_chunks, AudioSegment.empty())


def generate_audio(text_chunks, audio_path):
    full_audio = AudioSegment.empty()
    for i, chunk in enumerate(text_chunks):
        temp_path = f"{audio_path}.temp{i}.wav"
        try:
            tts.tts_to_file(
                text=chunk,
                file_path=temp_path,
                speaker=tts.speakers[0] if tts.speakers else None,
                language=tts.languages[0] if tts.languages else None,
                speed=0.95,  # Slower rate for better clarity
                emotion="Neutral",
                sample_rate=44100,
            )
            chunk_audio = AudioSegment.from_wav(temp_path)

            # Handle mumbles and strange noises
            chunk_audio = remove_strange_sounds(chunk_audio)

            if len(chunk_audio) > 0:
                full_audio += chunk_audio + AudioSegment.silent(duration=300)  # Reduced silence between chunks
            os.remove(temp_path)
        except Exception as e:
            logging.error(f"Error generating audio for chunk {i}: {str(e)}")

    full_audio.export(audio_path, format="wav")
    return audio_path


def post_process_audio(audio_path):
    audio = AudioSegment.from_wav(audio_path)

    # Normalize audio
    audio = audio.normalize()

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
                # Generate audio with enhanced settings
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
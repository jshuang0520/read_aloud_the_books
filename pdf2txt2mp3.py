import fitz  # PyMuPDF
from TTS.api import TTS
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
import numpy as np
from scipy.signal import butter, lfilter
from transformers import pipeline

nltk.download('punkt', quiet=True)

# Initialize sentiment analysis pipeline
nlp = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

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


def extract_text_from_pdf(page):
    text = page.get_text("text")
    # Remove any non-printable characters
    text = ''.join(filter(lambda x: x in string.printable, text))
    return text


def is_valid_text(text):
    result = nlp(text)[0]
    return result['score'] > 0.7  # Adjust threshold as needed


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

    # Remove duplicate punctuation marks
    text = re.sub(r'([.,?!;:])\1+', r'\1', text)

    # Handle quotes more carefully
    # Step 1: Remove ending quotes after sentence-ending punctuation (., !, or ?)
    text = re.sub(r'(?<=[.!?])\s*["\']', '', text)  # Fixing mismatched single and double quotes in regex
    # Step 2: Remove starting quotes before a capital letter (assumes this starts a new sentence)
    text = re.sub(r'["\'](?=\s*[A-Z])', '', text)  # Fixes regex to match starting quotes before a capitalized word

    # Handle hyphenated words
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)  # Join words split across lines
    text = re.sub(r'(\w+)-(\w+)', lambda m: f"{m.group(1)} {m.group(2)}" if m.group(1).lower() in [
        'three', 'minute', 'axis'] else m.group(0), text)

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
        for i in range(0, len(words), 10):
            chunk = ' '.join(words[i:i + 10])
            if chunk:
                processed_chunks.append(chunk)

    # Final processing of chunks
    final_chunks = []
    for chunk in processed_chunks:
        chunk = chunk.capitalize()
        chunk = ''.join(filter(lambda x: x in string.printable, chunk))
        # Check if the chunk contains a minimum percentage of alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in chunk) / len(chunk) if len(chunk) > 0 else 0
        if chunk.strip() and not chunk.isspace() and len(chunk.split()) >= 3 and alpha_ratio > 0.5 and is_valid_text(
                chunk):
            final_chunks.append(chunk)

    return final_chunks


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def check_audio_quality(audio_segment):
    # Check for very short segments
    if len(audio_segment) < 500:  # Less than 500ms
        return False

    # Check for segments with very low volume
    if audio_segment.dBFS < -30:
        return False

    return True


def remove_strange_sounds(audio):
    # Split audio on silence
    chunks = split_on_silence(audio, min_silence_len=100, silence_thresh=-40)

    good_chunks = []
    for chunk in chunks:
        if len(chunk) > 100 and chunk.dBFS > -35:
            sub_chunks = split_on_silence(chunk, min_silence_len=50, silence_thresh=-35)
            if all(abs(c.dBFS - chunk.dBFS) < 15 for c in sub_chunks):
                # Apply bandpass filter
                samples = np.array(chunk.get_array_of_samples())
                filtered = bandpass_filter(samples, lowcut=300, highcut=3000, fs=chunk.frame_rate)
                filtered_chunk = AudioSegment(
                    filtered.astype(np.int16).tobytes(),
                    frame_rate=chunk.frame_rate,
                    sample_width=2,
                    channels=1
                )
                if check_audio_quality(filtered_chunk):
                    good_chunks.append(filtered_chunk)
            else:
                good_chunks.extend([c for c in sub_chunks if len(c) > 200 and check_audio_quality(c)])

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
                speed=0.95,
                emotion="Neutral",
                sample_rate=44100,
            )
            chunk_audio = AudioSegment.from_wav(temp_path)
            chunk_audio = remove_strange_sounds(chunk_audio)

            if len(chunk_audio) > 0 and check_audio_quality(chunk_audio):
                full_audio += chunk_audio + AudioSegment.silent(duration=500)
            else:
                logging.warning(f"Discarded low-quality audio for chunk: {chunk[:30]}...")
            os.remove(temp_path)
        except Exception as e:
            logging.error(f"Error generating audio for chunk {i}: {str(e)}")

    full_audio = normalize(full_audio)
    full_audio.export(audio_path, format="wav")
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

    # Apply spectral subtraction (simplified version)
    noise_sample = audio[:1000]  # Assume first 1 second is noise
    noise_spectrum = np.fft.fft(noise_sample.get_array_of_samples())
    audio_spectrum = np.fft.fft(audio.get_array_of_samples())
    clean_spectrum = audio_spectrum - noise_spectrum[:len(audio_spectrum)]
    clean_audio = np.fft.ifft(clean_spectrum).real.astype(np.int16)

    # Convert back to AudioSegment
    clean_audio_segment = AudioSegment(
        clean_audio.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )

    # Export processed audio
    clean_audio_segment.export(audio_path, format="wav")


def process_batch(start_page, end_page):
    doc = fitz.open(pdf_path)
    for page_num in range(start_page, end_page):
        # if (page_num == 268) or (page_num == 269) or (page_num == 272):  # these are marked as the pages with strange, creepy noises / mumbles
        page = doc[page_num]
        text = extract_text_from_pdf(page)
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


"""
> prompt
I tried TTS, the package to generate audio files from text files, but it makes strange and creepy noise and mumbles even if I made efforts to text preprocessing, such as generate in a relative low speed, setting thresholds and cleaning sentences, getting rid of silence part, making long sentences to shorter one, etc. How should I do to completely get rid of the strange and creepy noise and mumbles? using other models or focusing on which of the data preprocessing or post-processing part?

> ans
Based on your experience and the search results, there are several approaches you can take to improve the quality of your TTS output and reduce strange noises and mumbles:
Try different TTS models:
The quality of TTS output can vary significantly between models. Consider experimenting with other models available in the TTS package or even other TTS libraries. For example, you could try:
Different models within TTS (e.g., FastPitch, Tacotron2, VITS)
Other libraries like gTTS, pyttsx3, or even cloud-based services like Google Cloud TTS or Amazon Polly
Focus on text preprocessing:
Enhance your text preprocessing steps 13:
Normalize non-standard words (NSWs) and abbreviations
Improve sentence segmentation
Implement more robust part-of-speech (POS) tagging
Enhance grapheme-to-phoneme (G2P) conversion
Implement proper name and acronym identification
Address pronunciation ambiguities, especially for homophones and homographs
Improve audio preprocessing:
Ensure proper resampling of your audio data to match the model's expected sample rate 4
Apply appropriate filtering to remove noise or unwanted frequencies
Enhance post-processing:
Implement more sophisticated silence removal techniques
Apply audio normalization to ensure consistent volume levels
Use spectral subtraction or other noise reduction techniques
Fine-tune model parameters:
Adjust the speaking rate, pitch, and other model-specific parameters
Experiment with different inference settings (e.g., temperature, top-k sampling)
Address specific issues:
For creepy noises: These might be caused by incorrect phoneme predictions or prosody issues. Focus on improving G2P conversion and prosody prediction in your preprocessing.
For mumbles: These could be due to unclear pronunciation or incorrect emphasis. Enhance your stress assignment and pronunciation disambiguation in preprocessing.
Implement a hybrid approach:
Combine statistical and rule-based methods for various preprocessing steps, as suggested in 2. This can help address complex linguistic nuances that pure statistical methods might miss.
Use supplementary data:
If available, utilize additional data like pitch information, speaker ID, or emotion labels to improve the TTS output quality 1.
Implement robust error handling:
Ensure your preprocessing pipeline can handle edge cases and unusual text inputs gracefully.
Consider domain-specific adaptations:
If your text is from a specific domain (e.g., medical, technical), adapt your preprocessing to handle domain-specific terminology and structures.
Remember that achieving high-quality TTS output often requires iterative refinement and a combination of techniques. Focus on systematically addressing each aspect of the pipeline, from text preprocessing to model selection and audio post-processing, to achieve the best results.


---


> prompt
inspired from your advise, I want to add more, generalized rules to preprocess the text data to this already built function:
def preprocess_text(text):
1. Normalize non-standard words and abbreviations
e.g. Dr. -> turn it into: "doctor"
2. replace some abbreviation of symbols
e.g. "Tip # 3" -> turn it into: "tip number three"
3. deal with different fonts well, e.g. bold fonts, fonts with headings, ...
4. deal with punctuation marks
e.g. Hyphen and Dash (-, _, etc.), and Apostrophe (he's -> he is)

> ans
This enhanced version of preprocess_text includes the following additional features:
Normalization of non-standard words and abbreviations:
Common titles (Dr., Mr., Mrs., Prof.) are expanded.
Common abbreviations (e.g., i.e., etc.) are expanded.
Replacement of symbol abbreviations:
'#' is replaced with 'number'.
'%' is replaced with 'percent'.
'&' is replaced with 'and'.
Number to word conversion:
Uses num2words library to convert numbers to their word representation.
Handling of apostrophes:
Contractions are expanded (e.g., "he's" -> "he is", "don't" -> "do not").
Handling of hyphens and dashes:
Hyphenated words are split (e.g., "well-known" -> "well known").
Note that this function doesn't directly handle different fonts or formatting (bold, headings, etc.) because these are typically not present in plain text. If you're extracting text from a formatted document (like a PDF or HTML), you'll need to handle those separately before passing the text to this function.
Also, keep in mind that this function might need further refinement depending on your specific use case and the nature of your input text. You may need to add more rules or adjust existing ones based on the patterns you observe in your data.

---


> prompt
I found that in these contexts:
3. Eliminate clutter. Identify elements that don’t add informative
value and remove them from your visuals. Leverage the Gestalt
principles to understand how people see and identify candidates
for elimination. Use contrast strategically. Employ alignment of
elements and maintain white space to help make the interpreta-
tion of your visuals a comfortable experience for your audience.
4. Focus attention where you want it. Employ the power of pre-
attentive attributes like color, size, and position to signal what’s
important. Use these strategic attributes to draw attention to
where you want your audience to look and guide your audience
through your visual. Evaluate the effectiveness of preattentive
attributes in your visual by applying the “where are your eyes
drawn?” test.
5. Think like a designer. Offer your audience visual affordances as
cues for how to interact with your communication: highlight the
important stuff, eliminate distractions, and create a visual hier-
archy of information. Make your designs accessible by not over-
complicating and leveraging text to label and explain. Increase
your audience’s tolerance of design issues by making your visu-
als aesthetically pleasing. Work to gain audience acceptance of
your visual designs.

the current function does not prevent it from generating strange noises and mumbles

I have some ideas to fix this, but if you have other good methods, feel free to add the methods in!
1. there are still long sentences even after the split, maybe we should shorter the sentences into like at most 10 words, to divide and conquer
2. there are some unfinished words, such as "stor- ytelling" it should be "stroytelling" in the regular way, but this can be challenge because there are still words with dashes, like "3-minute", "y-axis" means "three minutes" and "y axis", respectively, but we should figure this out
3. we should Improve sentence segmentation
4. Enhance grapheme-to-phoneme (G2P) conversion
5. Address pronunciation ambiguities, especially for homophones and homographs
6. Improve audio preprocessing:
- Ensure proper resampling of your audio data to match the model's expected sample rate
- Apply appropriate filtering to remove noise or unwanted frequencies
7. Enhance post-processing:
- Implement more sophisticated silence removal techniques
- Apply audio normalization to ensure consistent volume levels
- Use spectral subtraction or other noise reduction techniques
8. address mumbles: These could be due to unclear pronunciation or incorrect emphasis. Enhance your stress assignment and pronunciation disambiguation in preprocessing.
9. Implement robust error handling:
Ensure your preprocessing pipeline can handle edge cases and unusual text inputs gracefully

> ans
These changes address several of the issues you mentioned:
Shorter sentences: The preprocess_text function now splits sentences into chunks of at most 10 words.
Handling hyphenated words: Added logic to join words split across lines and handle specific cases like "three-minute" and "y-axis".
Improved sentence segmentation: Using NLTK's sent_tokenize for better sentence splitting.
Enhanced audio processing:
Added a bandpass filter to remove frequencies outside the typical speech range.
Implemented spectral subtraction for noise reduction.
Improved normalization and silence removal.
Robust error handling: The generate_audio function now has a try-except block for each chunk.
To further improve the script:
G2P conversion: Consider integrating a G2P library like g2p_en for better pronunciation handling.
Pronunciation ambiguities: You might want to create a custom dictionary for words with ambiguous pronunciations in your specific context.
Stress assignment: This is typically handled by the TTS model, but you could potentially add stress markers to important words in the preprocessing step.


---


> prompt
yes, awesome! 
well, in the preprocess part, I believe we can do better to deal with the following issues:
1. sentence with multiple punctuation marks: "“where are your eyes drawn?”" -> should be simplified to "where are your eyes drawn"
2. and maybe some of the words are not so clearly printed on the book, it may be challenging to parse the words correctly, maybe we should deal with it
3. while it is great, well, some of the voice are different from the original one, it seems like the original challenging parts are transformed into a radio station sound, which is not quite as clear as the original sound, maybe we can do some efforts to make these in a more alignment way
4. maybe it's because of our shorter sentences policy, some pauses are strange, like in half of a sentence, so can we make it more naturally?


> ans
This updated version incorporates the following improvements:
Handling multiple punctuation marks and quotes.
Spell checking to deal with potentially unclear words.
Improved sentence splitting for more natural pauses.
Increased pause duration between sentences for better rhythm.
Bandpass filtering to remove frequencies outside the typical speech range.
Spectral subtraction for noise reduction.
This version should address the issues you mentioned, including handling multiple punctuation marks, dealing with unclear words, improving the naturalness of pauses, and reducing strange noises. The audio quality and clarity should be improved, with a more consistent voice throughout the generated audio.
You may need to fine-tune some parameters (like silence durations, frequency cutoffs, etc.) based on your specific content and desired output. Also, keep in mind that the effectiveness of these improvements may vary depending on the specific characteristics of your input text and the TTS model being used.


---


> prompt
awesome, this version works really great!
but it seems like there are some sentences that are even don't exist, and the machine tries to generate audios from them, so eventually there exist some strange/noisy, creepy sounds that I have no clues they should be mapped to which of the sentences or contexts from the book
please provide me solutions whether preprocessing or postprocessing to deal with this issue, thank you


> ans
This updated version incorporates the following improvements:
Enhanced text extraction from PDF using extract_text_from_pdf function.
Improved text preprocessing with more rigorous checks for valid text.
Added a sentiment analysis check using the transformers library to filter out potentially nonsensical text.
Implemented an audio quality check function to discard low-quality audio segments.
Adjusted the silence duration between chunks for more natural pauses.
Enhanced error handling and logging throughout the process.

This version should help reduce the occurrence of strange noises, creepy sounds, and non-existent sentences in the generated audio. It implements more stringent checks on both the text and audio levels to ensure higher quality output.
You may need to fine-tune some parameters (like the sentiment analysis threshold, audio quality check thresholds, etc.) based on your specific content and desired output quality. Monitor the logs and output to adjust these parameters as needed.

"""
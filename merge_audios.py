from pydub import AudioSegment
import os
from TTS.api import TTS
import glob
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")


def generate_page_announcement(page_num, output_path):
    text = f"Page {page_num}"
    tts.tts_to_file(text=text, file_path=output_path)


def merge_and_concatenate(input_folder, output_file):
    # Get all .wav files and sort them
    input_files = sorted(glob.glob(os.path.join(input_folder, 'page_*.wav')))
    logging.info(f"Found {len(input_files)} input files under path: {input_folder}")

    final_audio = AudioSegment.empty()

    for file in input_files:
        if not os.path.exists(file):
            logging.warning(f"File not found: {file}")
            continue

        if os.path.getsize(file) == 0:
            logging.warning(f"Empty file: {file}")
            continue

        page_num = int(os.path.basename(file).split('_')[1].split('.')[0])
        logging.info(f"Processing page {page_num}")

        try:
            # Generate page announcement
            announcement_file = f"announcement_{page_num}.wav"
            generate_page_announcement(page_num, announcement_file)

            # Load the announcement and page content
            announcement = AudioSegment.from_wav(announcement_file)
            page_content = AudioSegment.from_wav(file)

            # Merge announcement and page content
            merged = announcement + AudioSegment.silent(duration=500) + page_content + AudioSegment.silent(duration=1000)

            # Add to final audio
            final_audio += merged

            # Clean up temporary announcement file
            os.remove(announcement_file)

        except Exception as e:
            logging.error(f"Error processing file {file}: {str(e)}")

    if len(final_audio) == 0:
        logging.error("No audio content to merge")
        return

    # Export the final audio file
    try:
        final_audio.export(output_file, format="wav")
        logging.info(f"Successfully exported merged audio to {output_file}")
    except Exception as e:
        logging.error(f"Error exporting final audio: {str(e)}")


# Usage
file_name = 'Storytelling_with_Data'
input_folder = "/Users/johnson.huang/py_ds/read_aloud_the_books/audio_output/Storytelling_with_Data-facebook-mms-tts-eng"
output_file = f"complete_audiobook_{file_name}.wav"

merge_and_concatenate(input_folder, output_file)

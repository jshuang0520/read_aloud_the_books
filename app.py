from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import PyPDF2
import pyttsx3
import threading
import time
import multiprocessing
import os
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)
socketio = SocketIO(app)
UPLOAD_FOLDER = "/Users/johnson.huang/py_ds/read_aloud_the_books/books"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

logging.basicConfig(level=logging.INFO)

bookmark = {"page": 0, "position": 0}
stop_flag = threading.Event()
current_position = 0

def extract_text_from_pdf(pdf_path, start_page=0):
    logging.info("Opening the PDF file...")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text_data = []
        num_pages = len(reader.pages)
        if start_page >= num_pages:
            raise ValueError(f"Start page {start_page} exceeds total pages {num_pages}.")
        
        for page_num in range(start_page, num_pages):
            logging.info(f"Extracting text from page {page_num + 1}...")
            page_text = reader.pages[page_num].extract_text()
            if page_text:
                # Add page markers for easy readability
                page_text = f"Page {page_num + 1}:\n{page_text}\n---\n"
                text_data.append(page_text)
            else:
                logging.warning(f"No text extracted from page {page_num + 1}. It may be empty or an image-only page.")
    
    logging.info("Finished extracting text from PDF.")
    return "\n".join(text_data)

def read_aloud_text(text_data, start_position=0, rate=150):
    logging.info("Initializing text-to-speech engine...")
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    words = text_data.split(" ")
    for i in range(start_position, len(words)):
        if stop_flag.is_set():
            logging.info("Stopping the text-to-speech process.")
            break
        current_position = i
        current_word = words[i]
        socketio.emit('highlight_word', {'position': i})
        engine.say(current_word)
        engine.runAndWait()
        time.sleep(0.05)
    logging.info("Finished reading aloud.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        return jsonify({"file_path": file_path})

@app.route("/load_text", methods=["POST"])
def load_text():
    pdf_path = request.form["pdf_path"]
    start_page = int(request.form.get("start_page", 0))
    rate = int(request.form.get("rate", 150))
    try:
        text_data = extract_text_from_pdf(pdf_path, start_page)
        return jsonify({"text_data": text_data, "start_position": bookmark["position"], "rate": rate})
    except FileNotFoundError:
        return jsonify({"error": f"File {pdf_path} not found. Please provide a valid PDF file."}), 404

@app.route("/bookmark", methods=["POST"])
def save_bookmark():
    global bookmark
    bookmark = request.json
    return jsonify({"status": "Bookmark saved!"})

@socketio.on('control')
def handle_control(data):
    global current_position
    command = data['command']
    rate = data.get('rate', 150)
    if command == 'play':
        stop_flag.clear()
        text_data = data['text_data']
        # Corrected to pass arguments directly without using args keyword
        socketio.start_background_task(run_tts_process, text_data, current_position, rate)
    elif command == 'pause':
        stop_flag.set()
        logging.info("Paused the reading.")
    elif command == 'stop':
        stop_flag.set()
        emit('reset_position', {'position': 0})
    elif command == 'go_back':
        current_position = max(0, current_position - int(5 * rate / 60))
        emit('reset_position', {'position': current_position})
    elif command == 'go_forward':
        current_position = min(len(data['text_data'].split(" ")) - 1, current_position + int(5 * rate / 60))
        emit('reset_position', {'position': current_position})

def run_tts_process(text_data, start_position, rate):
    process = multiprocessing.Process(target=read_aloud_text, args=(text_data, start_position, rate))
    process.start()
    process.join()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=True)


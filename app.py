from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import PyPDF2
import threading
import time
import os
from gtts import gTTS
from playsound import playsound  # For cross-platform MP3 playback

app = Flask(__name__)
socketio = SocketIO(app)
bookmark = {"page": 0, "position": 0}
stop_flag = threading.Event()
current_position = 0

def extract_text_from_pdf(pdf_path, start_page=0):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text_data = []
        num_pages = len(reader.pages)
        for page_num in range(start_page, num_pages):
            page_text = reader.pages[page_num].extract_text()
            page_text = f"--- Page {page_num + 1} ---\n" + page_text  # Add page separator
            text_data.append(page_text)
    return "\n".join(text_data)

def send_progress_update(total_words, current_word_index):
    progress = int((current_word_index / total_words) * 100)
    socketio.emit('update_progress', {'progress': progress})

def read_aloud_text_gtts(text_data, start_position=0):
    words = text_data.split(" ")
    total_words = len(words)

    for i in range(start_position, total_words):
        if stop_flag.is_set():
            break
        current_word = words[i]

        if current_word.strip():  # Ensure there's text to speak
            tts = gTTS(current_word, lang='en')
            tts.save('/tmp/current_word.mp3')
            playsound('/tmp/current_word.mp3')  # Play using playsound

        socketio.emit('highlight_word', {'position': i})
        send_progress_update(total_words, i)
        time.sleep(0.05)

@app.route("/")
def index():
    return render_template("index.html")

UPLOAD_FOLDER = "/Users/johnson.huang/py_ds/read_aloud_the_books"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if file:
        filename = file.filename
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        return jsonify({"file_path": file_path})

@app.route("/load_text", methods=["POST"])
def load_text():
    pdf_path = request.form["pdf_path"]
    start_page = int(request.form.get("start_page", 0))
    try:
        text_data = extract_text_from_pdf(pdf_path, start_page)
        return jsonify({"text_data": text_data, "start_position": bookmark["position"]})
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

    if command == 'initialize':
        stop_flag.clear()
        current_position = data.get('start_position', 0)
        text_data = data['text_data']
        if text_data.strip():  # Check that text is not empty
            socketio.start_background_task(target=read_aloud_text_gtts, text_data=text_data, start_position=current_position)
        else:
            emit('status', {'message': 'No text to speak!'})

    elif command == 'resume':
        stop_flag.clear()
        text_data = data['text_data']
        if text_data.strip():
            socketio.start_background_task(target=read_aloud_text_gtts, text_data=text_data, start_position=current_position)
        else:
            emit('status', {'message': 'No text to resume!'})

    elif command == 'pause':
        stop_flag.set()

    elif command == 'stop':
        stop_flag.set()
        emit('reset_position', {'position': 0})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8080, debug=True)


import os, io, requests, numpy as np
from flask import Flask, request
from PIL import Image
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import gdown

load_dotenv()  # load environment variables from .env if present

# Environment variables
TOKEN = os.environ.get('TELEGRAM_TOKEN')
MODEL_PATH = os.environ.get('MODEL_PATH', 'seed_detector.h5')
LABELS = os.environ.get('LABELS', 'board_bean,green_lentils,pea_seed,peppar_seed').split(',')
WEBHOOK_SECRET = os.environ.get('WEBHOOK_SECRET', 'abc123')

# Google Drive file ID for the model
GOOGLE_DRIVE_FILE_ID = "1QIY9fQnuSBIMXaEHydCn_ODnEDbNr-qu"
GDRIVE_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}&export=download"

# Ensure model file is present
if not os.path.exists(MODEL_PATH):
    print("⚙️ Model file not found locally. Downloading from Google Drive...")
    # output = MODEL_PATH, quiet = False so we see progress
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    print("✅ Model downloaded")

# Load model once at start-up
print(f"Loading model from {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print("✅ Model loaded")

app = Flask(__name__)

@app.route("/")
def index():
    return "Seed bot is alive!"

@app.route(f"/webhook/{WEBHOOK_SECRET}", methods=['POST'])
def webhook():
    data = request.get_json(force=True)
    try:
        message = data.get('message') or data.get('edited_message')
        if not message:
            return "ok", 200

        chat_id = message['chat']['id']

        # If photo sent
        if 'photo' in message:
            file_id = message['photo'][-1]['file_id']
            # Get file info
            r = requests.get(
                f"https://api.telegram.org/bot{TOKEN}/getFile?file_id={file_id}"
            ).json()
            file_path = r['result']['file_path']

            # Download the image
            file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
            img_resp = requests.get(file_url)

            # Preprocess image
            img = Image.open(io.BytesIO(img_resp.content)).convert("RGB").resize((224,224))
            x = np.expand_dims(np.array(img) / 255.0, 0)

            # Predict
            pred = model.predict(x)
            idx = int(pred.argmax())
            conf = float(pred[0][idx])
            text = f"{LABELS[idx]} ({conf*100:.1f}%)"

            # Send reply
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                json={'chat_id': chat_id, 'text': text}
            )
    except Exception as e:
        print("❌ Error:", e)

    return "ok", 200

if __name__ == "__main__":
    # Port from env var, default 5000
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

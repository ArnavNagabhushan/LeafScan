from flask import Flask, request, render_template, jsonify, redirect, url_for
import os, time, numpy as np, gdown, matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
GRAPH_FOLDER = 'static/graphs'
MODEL_PATH = 'model/plant_model.h5'
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

for folder in [UPLOAD_FOLDER, GRAPH_FOLDER, 'model']:
    os.makedirs(folder, exist_ok=True)

MODEL_ID = '1ANsJOrCuHeUIB3jVwbCstVcIakak08_e'
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
labels = ['Healthy', 'Disease1', 'Disease2']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    label = labels[idx]
    conf = round(float(preds[0][idx]) * 100, 2)
    rec = "Water regularly and treat disease if needed."

    plt.figure(figsize=(5, 3))
    bars = plt.bar(labels, preds[0] * 100, color='lightgreen')
    bars[idx].set_color('#004d00')
    plt.ylabel('Confidence %'); plt.title('Prediction Confidence'); plt.ylim(0, 100)
    gpath = os.path.join(GRAPH_FOLDER, 'plot.png')
    plt.savefig(gpath); plt.close()
    return label, conf, rec, gpath

def home():
    if request.method == 'POST':
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # 1️⃣ Get uploaded file
        file = request.files['file']
        if not file:
            return "No file uploaded", 400

        # 2️⃣ Save it temporarily
        filepath = os.path.join('static', 'uploads', file.filename)
        file.save(filepath)

        # 3️⃣ Load model (download from Drive if needed)
        model_path = "model.h5"
        if not os.path.exists(model_path):
            gdown.download("https://drive.google.com/uc?id=1ANsJOrCuHeUIB3jVwbCstVcIakak08_e", model_path, quiet=False)

        model = tf.keras.models.load_model(model_path)

        # 4️⃣ Preprocess the image
        img = cv2.imread(filepath)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # 5️⃣ Make prediction
        pred = model.predict(img)
        result = np.argmax(pred)

        # 6️⃣ Render result page
        return render_template('result.html', result=result, image=file.filename)

    return render_template('index.html')

@app.route('/how')
def how():
    return render_template('how.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        msg = request.form.get('message')
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {msg}\n---\n")
        return render_template('index.html', message=f"✅ Thanks, {name}!")
    return render_template('feedback.html')

@app.route('/results')
def results():
    return render_template('results.html',
        label=request.args.get('label'),
        confidence=request.args.get('confidence'),
        recommendation=request.args.get('recommendation'),
        filename=request.args.get('filename'),
        graph_path=request.args.get('graph_path')
    )

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected'})
    if not allowed_file(file.filename):
        return jsonify({'error': 'Allowed: png, jpg, jpeg'})

    fname = secure_filename(file.filename)
    fpath = os.path.join(UPLOAD_FOLDER, fname)
    file.save(fpath)
    label, conf, rec, gpath = model_predict(fpath)
    return redirect(url_for('results', label=label, confidence=conf,
                            recommendation=rec, filename=fname, graph_path=gpath))

@app.errorhandler(404)
def not_found(e):
    return redirect(url_for('home'))

if __name__ == '__main__':
    time.sleep(5)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
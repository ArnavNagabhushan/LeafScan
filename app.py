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

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)
os.makedirs('model', exist_ok=True)

# Auto-download model from Google Drive if missing
MODEL_ID = '1ANsJOrCuHeUIB3jVwbCstVcIakak08_e'
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Drive...")
    gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_PATH, quiet=False)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model ready ✅")

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
    rec = "Take proper care: water regularly, provide nutrients, and treat disease if needed."

    plt.figure(figsize=(5, 3))
    bars = plt.bar(labels, preds[0] * 100, color='lightgreen')
    bars[idx].set_color('#004d00')
    plt.ylabel('Confidence %'); plt.title('Prediction Confidence'); plt.ylim(0, 100)
    path = os.path.join(GRAPH_FOLDER, 'plot.png')
    plt.savefig(path); plt.close()
    return label, conf, rec, path

@app.route('/')
def home(): return render_template('index.html')

@app.route('/how')
def how(): return render_template('how.html')

@app.route('/faq')
def faq(): return render_template('faq.html')

@app.route('/team')
def team(): return render_template('team.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name, email, msg = request.form.get('name'), request.form.get('email'), request.form.get('message')
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {msg}\n---\n")
        return render_template('index.html', message=f"✅ Thanks for your feedback, {name}!")
    return render_template('feedback.html')

@app.route('/results')
def results():
    return render_template('results.html',
        label=request.args.get('label', 'Healthy'),
        confidence=request.args.get('confidence', 100),
        recommendation=request.args.get('recommendation', 'Take care of your plant!'),
        filename=request.args.get('filename', 'default_leaf.png'),
        graph_path=request.args.get('graph_path', 'static/graphs/plot.png')
    )

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'})
    if file and allowed_file(file.filename):
        fname = secure_filename(file.filename)
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        file.save(fpath)
        label, conf, rec, gpath = model_predict(fpath)
        return redirect(url_for('results', label=label, confidence=conf, recommendation=rec, filename=fname, graph_path=gpath))
    return jsonify({'error': 'Allowed file types: png, jpg, jpeg'})

@app.errorhandler(404)
def not_found(e): return redirect(url_for('home'))

if __name__ == '__main__':
    time.sleep(5)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model/plant_model.h5")

classes = [
    'Apple Scab', 'Apple Black Rot', 'Cedar Apple Rust', 'Healthy Apple',
    'Potato Early Blight', 'Potato Late Blight', 'Healthy Potato',
    'Tomato Early Blight', 'Tomato Late Blight', 'Healthy Tomato',
    'Corn Leaf Spot', 'Healthy Corn'
]

treatments = {
    'Apple Scab': "Use fungicides such as captan, mancozeb, or myclobutanil. Remove fallen leaves and prune infected areas.",
    'Apple Black Rot': "Apply copper-based fungicides. Remove infected branches and cankers.",
    'Cedar Apple Rust': "Use myclobutanil or propiconazole. Remove nearby juniper hosts if possible.",
    'Healthy Apple': "No treatment needed. Maintain routine watering and pruning.",

    'Potato Early Blight': "Use fungicides like chlorothalonil or mancozeb. Improve soil drainage and avoid overhead watering.",
    'Potato Late Blight': "Apply copper-based fungicides immediately. Destroy severely infected plants.",
    'Healthy Potato': "No treatment required. Continue routine monitoring.",

    'Tomato Early Blight': "Use copper sprays or chlorothalonil. Improve airflow and avoid wetting leaves.",
    'Tomato Late Blight': "Apply copper fungicide. Remove severely infected plants to prevent spread.",
    'Healthy Tomato': "Plant is healthy. Keep monitoring moisture and airflow.",

    'Corn Leaf Spot': "Apply fungicides containing azoxystrobin or pyraclostrobin. Rotate crops and avoid dense planting.",
    'Healthy Corn': "No issues detected. Maintain standard care."
}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x)[0]
    num_classes = len(preds)

    # Ensure class labels match prediction length
    if len(classes) >= num_classes:
        labels = classes[:num_classes]
    else:
        labels = classes + [f"Class {i+1}" for i in range(len(classes), num_classes)]

    pred_idx = np.argmax(preds)
    label = labels[pred_idx]
    conf = round(preds[pred_idx]*100, 2)

    rec = treatments.get(label, "No specific recommendation available.")
    
    top_idx = preds.argsort()[-5:][::-1]
    top_preds = preds[top_idx]*100
    top_labels = [labels[i] for i in top_idx]

    plt.figure(figsize=(16,10))
    plt.bar(top_labels, top_preds, color='lightgreen')
    plt.xticks(rotation=75, ha='right', fontsize=12)
    plt.ylabel('Confidence (%)', fontsize=14)
    plt.title('Top Predictions', fontsize=16)
    plt.tight_layout()

    if not os.path.exists('static'):
        os.makedirs('static')
    graph_path = "static/prediction_graph.png"
    plt.savefig(graph_path)
    plt.close()

    return label, conf, rec, graph_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded!"
        file = request.files['file']
        if file.filename == '':
            return "No file selected!"

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        label, conf, rec, graph_path = model_predict(filepath)
        return render_template("results.html", label=label, conf=conf, rec=rec, graph_path=graph_path)
    
    return render_template("index.html")

@app.route('/faq')
def faq():
    return render_template("faq.html")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/how')
def how():
    return render_template("how.html")

@app.route('/feedback')
def feedback():
    return render_template("feedback.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
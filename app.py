# app.py - Updated version
from flask import Flask, render_template, request, redirect, url_for
import os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "model/plant_model.h5"
CLASSFILE = "real_class_names.txt"  # Updated to use correct file
UPLOAD_FOLDER = "uploads"
STATIC_GRAPH = "static/prediction_graph.png"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# Load class names
class_names = []
if os.path.exists(CLASSFILE):
    with open(CLASSFILE, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(class_names)} class names")
else:
    print("WARNING: No class_names file found!")

# Expanded treatment recommendations
treatments = {
    'Orange__Haunglongbing_(Citrus_greening)': "This is a serious bacterial disease. Remove infected trees and control psyllid insects. Consult agricultural experts.",
    'Peach__Bacterial_spot': "Apply copper-based bactericides. Prune infected branches and improve air circulation.",
    'Peach__healthy': "Your peach plant is healthy! Continue regular care and monitoring.",
    'Pepper,_bell__Bacterial_spot': "Use copper sprays and avoid overhead watering. Remove infected leaves.",
    'Pepper,_bell__healthy': "Bell pepper is healthy. Maintain good watering practices.",
    'Potato__Early_blight': "Use chlorothalonil fungicide. Improve airflow and avoid wetting foliage.",
    'Potato__Late_blight': "Apply copper sprays immediately. Remove severely affected leaves. This spreads quickly!",
    'Potato__healthy': "Potato plant is healthy. Continue normal care.",
    'Soybean__healthy': "Soybean is healthy. No treatment needed.",
    'Squash__Powdery_mildew': "Apply sulfur-based fungicides or neem oil. Improve air circulation.",
    'Strawberry__Leaf_scorch': "Remove infected leaves. Apply appropriate fungicides and ensure good drainage.",
    'Strawberry__healthy': "Strawberry plant is healthy. Continue regular maintenance.",
    'Tomato__Bacterial_spot': "Use copper-based sprays. Avoid overhead irrigation and remove infected parts.",
    'Tomato__Early_blight': "Apply chlorothalonil or copper fungicides. Mulch to prevent soil splash.",
    'Tomato__Late_blight': "Immediate copper treatment needed. This disease spreads rapidly!",
    'Tomato__Leaf_Mold': "Improve air circulation. Use fungicides if needed. Reduce humidity.",
    'Tomato__Septoria_leaf_spot': "Remove infected leaves. Apply fungicides and mulch around plants.",
    'Tomato__Spider-mites Two-spotted_spider_mite': "Use insecticidal soap or neem oil. Increase humidity around plants.",
    'Tomato__Target_Spot': "Apply fungicides containing chlorothalonil. Remove infected plant parts.",
    'Tomato__Tomato_Yellow_Leaf_Curl_Virus': "Control whitefly vectors. Remove infected plants to prevent spread.",
    'Tomato__Tomato_mosiac_virus': "No cure available. Remove infected plants. Disinfect tools and control aphids.",
    'Tomato__healthy': "Tomato plant is healthy! Continue good care practices."
}

def model_predict(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    # Get predictions
    preds = model.predict(x)[0]

    # Ensure probability-like output
    if not np.isclose(np.sum(preds), 1.0, atol=1e-2):
        preds = tf.nn.softmax(preds).numpy()

    # Get predicted class
    pred_idx = int(np.argmax(preds))
    label = class_names[pred_idx] if pred_idx < len(class_names) else f"Unknown_{pred_idx}"
    conf = round(float(preds[pred_idx]) * 100, 2)

    # Get recommendation
    rec = treatments.get(label, "No specific recommendation available. Consult a local agricultural expert.")

    # Generate graph (top 5)
    top_idx = preds.argsort()[-5:][::-1]
    top_labels = [class_names[i] if i < len(class_names) else f"Class_{i}" for i in top_idx]
    top_vals = preds[top_idx] * 100

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(top_labels)), top_vals, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6'])
    plt.xticks(range(len(top_labels)), top_labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Confidence (%)", fontsize=12)
    plt.title("Top 5 Predictions", fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_vals)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(STATIC_GRAPH, dpi=100, bbox_inches='tight')
    plt.close()

    return label, conf, rec, STATIC_GRAPH

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", error="Please select a file to upload.")

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        try:
            label, conf, rec, graph = model_predict(path)
            return render_template("results.html",
                                 label=label,
                                 conf=conf,
                                 rec=rec,
                                 graph_path="prediction_graph.png")
        except Exception as e:
            return render_template("index.html", error=f"Error processing image: {str(e)}")
    
    return render_template("index.html")

@app.route("/how")
def how():
    return render_template("how.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/team")
def team():
    return render_template("team.html")

@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    if request.method == "POST":
        return render_template("feedback.html", submitted=True)
    return render_template("feedback.html", submitted=False)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting LeafScan on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
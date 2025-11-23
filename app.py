from flask import Flask, render_template, request
import os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "model/plant_model.h5"
CLASSFILE = "real_class_names.txt"
UPLOAD_FOLDER = "uploads"
STATIC_GRAPH = "static/prediction_graph.png"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

model = load_model(MODEL_PATH, compile=False)

with open(CLASSFILE, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

treatments = {
    'Apple___Apple_scab': "Use fungicides like captan. Remove infected leaves.",
    'Apple___Black_rot': "Apply copper fungicides. Prune infected branches.",
    'Apple___Cedar_apple_rust': "Use myclobutanil fungicide.",
    'Apple___healthy': "Plant is healthy. Continue regular care.",
    'Blueberry___healthy': "Blueberry is healthy. Maintain acidic soil.",
    'Cherry_(including_sour)___Powdery_mildew': "Apply sulfur fungicides. Improve air circulation.",
    'Cherry_(including_sour)___healthy': "Cherry is healthy.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use azoxystrobin fungicide. Rotate crops.",
    'Corn_(maize)___Common_rust_': "Apply fungicide if severe. Use resistant varieties.",
    'Corn_(maize)___Northern_Leaf_Blight': "Remove infected leaves. Use resistant varieties.",
    'Corn_(maize)___healthy': "Corn is healthy.",
    'Grape___Black_rot': "Use mancozeb fungicide. Remove infected berries.",
    'Grape___Esca_(Black_Measles)': "Prune infected areas. Improve drainage.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Apply copper fungicides.",
    'Grape___healthy': "Grape vine is healthy.",
    'Orange___Haunglongbing_(Citrus_greening)': "Remove infected trees. Control psyllid insects.",
    'Peach___Bacterial_spot': "Apply copper bactericides. Improve air circulation.",
    'Peach___healthy': "Peach is healthy.",
    'Pepper,_bell___Bacterial_spot': "Use copper sprays. Avoid overhead watering.",
    'Pepper,_bell___healthy': "Bell pepper is healthy.",
    'Potato___Early_blight': "Use chlorothalonil fungicide. Improve airflow.",
    'Potato___Late_blight': "Apply copper sprays immediately. Remove affected leaves.",
    'Potato___healthy': "Potato is healthy.",
    'Raspberry___healthy': "Raspberry is healthy.",
    'Soybean___healthy': "Soybean is healthy.",
    'Squash___Powdery_mildew': "Apply sulfur fungicides or neem oil.",
    'Strawberry___Leaf_scorch': "Remove infected leaves. Ensure good drainage.",
    'Strawberry___healthy': "Strawberry is healthy.",
    'Tomato___Bacterial_spot': "Use copper sprays. Remove infected parts.",
    'Tomato___Early_blight': "Apply copper fungicides. Mulch to prevent splash.",
    'Tomato___Late_blight': "Immediate copper treatment needed. Disease spreads fast.",
    'Tomato___Leaf_Mold': "Improve air circulation. Reduce humidity.",
    'Tomato___Septoria_leaf_spot': "Remove infected leaves. Apply fungicides.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use insecticidal soap or neem oil.",
    'Tomato___Target_Spot': "Apply chlorothalonil fungicide.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control whiteflies. Remove infected plants.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants. Disinfect tools.",
    'Tomato___healthy': "Tomato is healthy."
}

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)[0]
    
    if not np.isclose(np.sum(preds), 1.0, atol=1e-2):
        preds = tf.nn.softmax(preds).numpy()

    pred_idx = int(np.argmax(preds))
    label = class_names[pred_idx]
    conf = round(float(preds[pred_idx]) * 100, 2)
    rec = treatments.get(label, "Consult agricultural expert for treatment advice.")

    top_idx = preds.argsort()[-5:][::-1]
    top_labels = [class_names[i] for i in top_idx]
    top_vals = preds[top_idx] * 100

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(top_labels)), top_vals, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6'])
    plt.xticks(range(len(top_labels)), top_labels, rotation=45, ha="right", fontsize=9)
    plt.ylabel("Confidence (%)", fontsize=12)
    plt.title("Top 5 Predictions", fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    
    for bar, val in zip(bars, top_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(STATIC_GRAPH, dpi=100, bbox_inches='tight')
    plt.close()

    return label, conf, rec

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", error="Please select a file.")

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        try:
            label, conf, rec = model_predict(path)
            return render_template("results.html", label=label, conf=conf, rec=rec)
        except Exception as e:
            return render_template("index.html", error=f"Error: {str(e)}")
    
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
    app.run(host="0.0.0.0", port=port, debug=False)
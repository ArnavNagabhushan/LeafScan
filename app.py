# -----------------------------------------------------------
# 🌿 LeafScan - Super Simple Leaf Disease Detection Web App
# Team: B1052JR2
# Backend Developer: Arnav Nagabhushan
# Purpose: Upload a leaf image → Get disease prediction
# Technology: Python, Flask, TensorFlow Keras, Matplotlib
# -----------------------------------------------------------

# ------------------- STEP 1 -------------------------------
# Importing all the required libraries for this project
# -----------------------------------------------------------
from flask import Flask, request, render_template, jsonify, redirect, url_for
import os                     # To work with directories and paths
import numpy as np            # For handling arrays and numbers
from tensorflow.keras.models import load_model  # Load Keras model
from tensorflow.keras.preprocessing import image # Preprocess images
import time                   # To add delays
import matplotlib.pyplot as plt # For plotting prediction confidence graphs
from werkzeug.utils import secure_filename  # Safely handle filenames

# ------------------- STEP 2 -------------------------------
# Setting up configuration for file uploads and models
# -----------------------------------------------------------
upload_folder = 'static/uploads'     # Folder where uploaded leaf images will be saved
graph_folder = 'static/graphs'       # Folder where prediction graphs will be saved
model_path = 'model/plant_model.h5'  # Path to the trained Keras model
allowed_extensions = {'png', 'jpg', 'jpeg'}  # Allowed file types for upload

# Ensure that folders exist
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(graph_folder, exist_ok=True)

# ------------------- STEP 3 -------------------------------
# Initialize the Flask application
# -----------------------------------------------------------
app = Flask(__name__)

# ------------------- STEP 4 -------------------------------
# Loading the trained Keras model into memory
# -----------------------------------------------------------
print("🔄 Loading model... please wait. This may take a few seconds...")
model = load_model(model_path)
print("✅ Model loaded successfully! Ready to predict leaf diseases!")

# ------------------- STEP 5 -------------------------------
# Defining the possible class labels for predictions
# -----------------------------------------------------------
# Replace 'Disease1', 'Disease2' with actual disease names your model predicts
labels = ['Healthy', 'Disease1', 'Disease2']

# ------------------- STEP 6 -------------------------------
# Helper function to check if the uploaded file has an allowed extension
# -----------------------------------------------------------
def allowed_file(filename):
    """
    Checks whether the uploaded file has a valid extension.

    Parameters:
        filename (str): Name of the uploaded file
    Returns:
        bool: True if allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

# ------------------- STEP 7 -------------------------------
# Helper function to preprocess image and predict disease
# -----------------------------------------------------------
def model_predict(img_path):
    """
    Takes an image path, preprocesses it, feeds it into the Keras model,
    returns predicted label, confidence, recommendation, and generates a bar graph.
    """
    # Load image and preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Predict
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    label = labels[class_idx]
    confidence = float(preds[0][class_idx]) * 100

    # Recommendation
    recommendation = "Take proper care: water regularly, provide nutrients, and treat disease if needed."

    # Plot confidence graph
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, preds[0] * 100, color='lightgreen')
    for bar, lab in zip(bars, labels):
        if lab == label:
            bar.set_color('#004d00')
    plt.ylabel('Confidence %')
    plt.title('Prediction Confidence')
    plt.ylim(0, 100)
    plt.tight_layout()
    graph_path = os.path.join(graph_folder, 'plot.png')
    plt.savefig(graph_path)
    plt.close()

    return label, round(confidence, 2), recommendation, graph_path

# ------------------- STEP 8 -------------------------------
# Home page route
# -----------------------------------------------------------
@app.route('/')
def index():
    """
    Render the main home page of LeafScan.
    """
    return render_template('index.html')

# ------------------- STEP 9 -------------------------------
# How it works page
# -----------------------------------------------------------
@app.route('/how')
def how():
    """
    Render 'How It Works' page.
    """
    return render_template('how.html')

# ------------------- STEP 10 ------------------------------
# FAQ page
# -----------------------------------------------------------
@app.route('/faq')
def faq():
    """
    Render FAQ page.
    """
    return render_template('faq.html')

# ------------------- STEP 11 ------------------------------
# Team page
# -----------------------------------------------------------
@app.route('/team')
def team():
    """
    Render Team page.
    """
    return render_template('team.html')

# ------------------- STEP 12 ------------------------------
# Feedback route (works from both feedback.html and index.html)
# -----------------------------------------------------------
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    """
    Render and process feedback form submissions.
    Supports POST (for form submission) and GET (for page display).
    """
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')

        # Log feedback to file
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Name: {name}\nEmail: {email}\nMessage: {message}\n---\n")

        print(f"💬 Feedback from {name} ({email}): {message}")

        # Show thank-you message on homepage
        return render_template('index.html', message=f"✅ Thank you for your feedback, {name}!")

    # If GET, show feedback page
    return render_template('feedback.html')

# ------------------- STEP 13 ------------------------------
# Results route
# -----------------------------------------------------------
@app.route('/results')
def results():
    """
    Display the results page with predicted info and confidence graph.
    """
    label = request.args.get('label', 'Healthy')
    confidence = request.args.get('confidence', 100)
    recommendation = request.args.get('recommendation', "Take proper care of your plant!")
    filename = request.args.get('filename', 'default_leaf.png')
    graph_path = request.args.get('graph_path', 'static/graphs/plot.png')

    return render_template(
        'results.html',
        label=label,
        confidence=confidence,
        recommendation=recommendation,
        filename=filename,
        graph_path=graph_path
    )

# ------------------- STEP 14 ------------------------------
# Predict route
# -----------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles leaf image upload and prediction using the trained model.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        # Predict using model
        label, confidence, recommendation, graph_path = model_predict(filepath)

        # Redirect to results page
        return redirect(url_for(
            'results',
            label=label,
            confidence=confidence,
            recommendation=recommendation,
            filename=filename,
            graph_path=graph_path
        ))
    else:
        return jsonify({'error': 'Allowed file types: png, jpg, jpeg'})

# ------------------- STEP 15 ------------------------------
# 404 handler
# -----------------------------------------------------------
@app.errorhandler(404)
def page_not_found(e):
    """
    Redirect invalid URLs to the home page.
    """
    return redirect(url_for('index'))

# ------------------- STEP 16 ------------------------------
# Run Flask app
# -----------------------------------------------------------
if __name__ == '__main__':
    time.sleep(15)
    print("🌱 Starting LeafScan server at http://127.0.0.1:5000")
    print("💡 Stop the server anytime with CTRL + C")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
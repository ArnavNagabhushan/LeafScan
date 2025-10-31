# 🌿 LeafScan – Detecting Plant Diseases, Leaf by Leaf

**LeafScan** is an AI-powered web application that detects plant diseases from leaf images using a deep learning model.  
Built by **Team B1052JR2** for the **AI VidyaSetu 1.0 – Code for New Bharat** hackathon.

---

## 🚀 Features

- Upload a plant leaf image and get instant predictions  
- Displays **disease name**, **confidence percentage**, and a **visual bar graph**  
- Simple, modern web interface built with **Flask + HTML/CSS/JS**  
- Trained using the **PlantVillage dataset**  
- Generates confidence visualization graphs using **Matplotlib**

---

## 📁 Project Structure

LeafScan/
│
├── README.md ← You are here!
├── app.py ← Flask backend
├── train_model.py ← Model training script
│
├── static/
│ ├── style.css
│ ├── script.js
│ └── logo.png
│
├── templates/
│ ├── index.html
│ ├── results.html
│ ├── faq.html
│ ├── feedback.html
│ ├── how.html
│ └── team.html
│
├── model/
│ └── plant_model.h5
│
└── dataset/
└── (PlantVillage dataset)

---

## 📸 Usage

1. Open the web app.  
2. Upload a clear image of a plant leaf (**JPG/PNG**).  
3. Click **Upload & Predict**.  
4. Wait for the AI to analyze the image.  
5. View the **predicted disease name** and **confidence percentage**.

---

## 🧬 Model Overview

The deep learning model used is a **Convolutional Neural Network (CNN)** trained on the **PlantVillage** dataset.

**Model Pipeline:**

Input Image → Resize (128x128) → CNN Layers → Flatten → Dense → Softmax Output

**Example Classes:**
- Healthy  
- Early Blight  
- Late Blight  
- Rust  
- Leaf Mold  
- (and more depending on dataset version)

---

## 🌍 Future Enhancements

- 🌐 Deploy on Render / Vercel / Hugging Face Spaces  
- 🗣️ Add voice assistance for accessibility  
- 🈯 Introduce multi-language support (Hindi, Kannada, etc.)  
- 📷 Integrate real-time camera capture  
- 📱 Create a mobile-friendly Progressive Web App (PWA)

---

## 👥 Team B1052JR2 – *Creators of LeafScan 🌿*

| 🧾 **Name** | 💼 **Role** | 🌟 **Special Contribution** |
|:-------------|:-------------|:-----------------------------|
| 🧠 **Arnav Nagabhushan** | 🧑‍💻 **Team Leader & Backend Developer** | Designed Flask backend, model integration, training & AI logic |
| 💻 **Pratyush** | 🎯 **Frontend Developer** | Built responsive and interactive UI with HTML, CSS & JS |
| 🎨 **Sudhanshu Bugalia** | 🖌️ **UI/UX Designer** | Created user-friendly layouts and visual themes |
| 📊 **Atharva Mishra** | 🧩 **Data Engineer** | Handled dataset preprocessing and model training |
| 🎤 **I. K. Dhanyashree** | 🗣️ **Presentation Lead** | Designed and presented final hackathon pitch |

---

🏫 **School:** *P.M. Shri Kendriya Vidyalaya No. 2, Jalahalli East*  
💡 **Hackathon:** *AI VidyaSetu 1.0 – Code for New Bharat*  
🚀 **Project:** *LeafScan – Detecting plant diseases, leaf by leaf*  
📅 **Year:** 2025  

---

> ✨ *“A small idea rooted in code can grow into something that saves millions of plants.”* 🌱

---

## 🏁 License

This project is **open-source** and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## 💬 Contact

📧 **Team LeafScan**  
For suggestions, collaborations, or feedback — feel free to reach out!  
Let’s grow a greener future together 🌱
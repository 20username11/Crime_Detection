# 🛡️ Crime Detection System using Deep Learning

## 📌 Overview
This project is an **AI-powered crime detection system** that uses **Deep Learning** to identify criminal activities from video footage in real time.  
The model can detect multiple crime categories, notify the admin when suspicious activity is detected, and provide an **interactive dashboard** for reviewing flagged videos.

The solution is built using:
- **TensorFlow/Keras** (DenseNet121 Transfer Learning)
- **OpenCV** (Video frame preprocessing)
- **Flask** (Web dashboard & backend)
- **UCF Crime Dataset** from Kaggle

**Dataset Link**: [UCF Crime Dataset on Kaggle](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset)

---

## 🎯 Features
- **Crime Classification**:
  - Detects multiple crime categories like Robbery, Shoplifting, Fighting, Burglary, Arson, Stealing, Assault, Arrest, Explosion, Shooting, Vandalism, Abuse, etc.
- **Real-Time Alerts**:
  - Admin dashboard displays an alert if suspicious activity is detected.
- **Video Review**:
  - Embedded video playback for flagged footage.
- **Send to Authorities**:
  - One-click button to send the detected video to the police.
- **Data Visualization**:
  - Graphs showing dataset distribution.
  - ROC Curve for model evaluation.

---

## 📂 Dataset
The project uses the **UCF Crime Dataset** from Kaggle, which contains labeled video frames for various crime and non-crime scenarios.

### Dataset Details:
- **Categories**: 14 crime types + normal videos.
- **Train/Test Split**:
  - 80% Training data
  - 20% Testing data
- Data was **cleaned, resized (64x64)**, and normalized for training.

---

## 🛠️ Tech Stack
- **Programming Language**: Python 3.x
- **Frameworks & Libraries**:
  - Flask
  - TensorFlow / Keras
  - OpenCV
  - NumPy, Pandas
  - Matplotlib, Seaborn, Plotly
- **Model Architecture**: DenseNet121 (Transfer Learning)

---

## 🧠 Model Architecture
- **Base Model**: DenseNet121 (pretrained on ImageNet)
- **Modifications**:
  - Global Average Pooling
  - Fully Connected Layers: 256 → 512 → 1024 units
  - Dropout regularization
  - Output layer with softmax activation (14 classes)
- **Optimizer**: Adam
- **Loss**: Categorical Crossentropy
- **Metric**: AUC (Area Under Curve)

---

## 📊 Results
- **Test ROC AUC**: ~0.812


## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/Crime_Detection.git
cd Crime_Detection
```
###2️⃣ Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3️⃣ Run the Flask App
```bash
python app.py
```

### The dashboard will be available at:
```bash 
http://127.0.0.1:5001/dashboard
```

### Project Structure
```plaintext
Crime_Detection/
│
├── app.py                         # Flask backend and dashboard
├── Crime_detection.ipynb          # Model training script
├── crime_detection_model.h5       # Trained model weights
├── requirements.txt               # Dependencies list
├── static/                        # Static files (CSS, videos, charts)
│   ├── AdobeStock_984294293_Video_HD_Preview.mp4
│   ├── roc_curve.png
│   └── data_distribution.png
└── templates/
    └── admin_dashboard.html       # Admin dashboard UI



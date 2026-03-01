# 🧠 AI Deepfake Image Detection

An end-to-end deepfake detection system that classifies facial images as REAL or FAKE using deep learning.

🚀 Features
- 🔍 Real vs Fake image classification  
- 🧠 Transfer Learning with MobileNetV2  
- 📊 Confidence score display  
- 🌐 Interactive Streamlit web app  
- ⚡ Real-time image prediction  

🧪 Model Details
- Base Model: MobileNetV2  
- Input Size: 224 × 224  
- Framework: TensorFlow / Keras  
- Accuracy: ~84–85%  
- Dataset: Real vs Fake face images  

📸 Demo
Upload an image in the Streamlit app to check whether it is REAL or FAKE.


🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- Streamlit  
- NumPy  

▶️ How to Run Locally
```bash
git clone https://github.com/Swatiprasad269/deepfake-image-detection.git
cd deepfake-image-detection
pip install -r requirements.txt
streamlit run app.py

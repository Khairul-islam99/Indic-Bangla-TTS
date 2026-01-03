# Indic Parler TTS API (Bengali) 🇧🇩

A high-performance FastAPI wrapper for the **Indic Parler TTS** model (`ai4bharat/indic-parler-tts`). This project provides a production-ready API for generating high-quality Bengali speech from text, optimized for long-form content with intelligent chunking, progress tracking, and real-time hardware diagnostics.

## 🛠️ Installation & Setup

Follow these steps strictly to set up the project on your local machine or server.

### 1. Clone the Repository
Download the project code to your machine.
```bash
git clone https://github.com/Khairul-islam99/Indic-Bangla-TTS.git
cd Indic-Bangla-TTS
```
### 2. Create Virtual Environment
It is recommended to use Conda to manage dependencies and avoid conflicts.
```bash
conda create -n indic-tts python=3.10 -y
conda activate indic-tts
```
## 3. Install PyTorch (Crucial Step)
You must install the PyTorch version that matches your hardware BEFORE installing other requirements. Run ONE of the following commands:

For NVIDIA GPU (CUDA 12.x) - Recommended Use this for modern GPUs (RTX 30xx, 40xx, A100, H100).
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
### 4. Install Dependencies
Now install the remaining Python libraries.
```bash
pip install -r requirements.txt
```
## 🚀 Usage
1. Start the Server
Run the main application file. The server will initialize the model (this may take a minute) and start listening for requests.
```bash
python main.py
```
2. API Documentation
Access the interactive Swagger UI to test endpoints manually:
 URL: http://localhost:8000/docs

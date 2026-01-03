import os
import uvicorn
import torch  # ✅ Added torch to check hardware status
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tts_engine import tts_engine
import config

# Workaround for Intel OpenMP runtime duplicate error on Windows environments.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize FastAPI application
app = FastAPI(
    title="Indic Parler TTS API",
    description="Professional Bengali Text-to-Speech API with fixed voice profiling.",
    version="1.0"
)

class TTSRequest(BaseModel):
    """
    Request model for the synthesis endpoint.
    Accepts only text; voice parameters are controlled via server configuration.
    """
    text: str

@app.get("/", tags=["Health"])
def health_check():
    """
    Health check endpoint.
    Returns service status and hardware (GPU/CPU) information.
    """
    # Check Hardware Status dynamically
    is_gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_gpu_available else "Standard CPU"

    return {
        "status": "active", 
        "service": "Indic Parler TTS API", 
        "voice_profile": "(Configured in config.py)",
        # ✅ Hardware Diagnostics added here
        "hardware": {
            "device": "cuda" if is_gpu_available else "cpu",
            "gpu_available": is_gpu_available,
            "gpu_name": device_name
        }
    }

@app.post("/synthesize/", tags=["Synthesis"])
def synthesize_audio(request: TTSRequest):
    """
    Primary endpoint for text-to-speech synthesis.
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Input text is required.")

    try:
        # Generate audio using the TTS engine
        audio_buffer = tts_engine.generate_audio(request.text)
        
        return StreamingResponse(
            audio_buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
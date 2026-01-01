import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from tts_engine import tts_engine
import config

# Workaround for Intel OpenMP runtime duplicate error on Windows environments.
# Ensures stability when multiple libraries link against OpenMP.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Indic Parler TTS API",
    description="Professional Bengali Text-to-Speech API with fixed voice profiling.",
    version="1.0.0"
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
    Health check endpoint to verify service status and configuration.
    """
    return {
        "status": "active", 
        "service": "Indic Parler TTS API", 
        "voice_profile": "ok"
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
        # The voice description is injected automatically from config.py within the engine
        audio_buffer = tts_engine.generate_audio(request.text)
        
        return StreamingResponse(
            audio_buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )
    except Exception as e:
        # Log critical errors and return a 500 status
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    # Start the Uvicorn server using settings from config.py
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
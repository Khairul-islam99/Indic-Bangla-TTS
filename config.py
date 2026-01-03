import os

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# The Hugging Face model repository ID.
MODEL_ID = "ai4bharat/indic-parler-tts"

# =============================================================================
# VOICE PROFILE CONFIGURATION
# =============================================================================
# The fixed voice prompt used for synthesis.
# "Rashmi" is selected for her clear, soft, and calm voice.
# Modification of this string will affect the voice style globally.
VOICE_DESCRIPTION = (
    "Rashmi speaks slowly with a soft, warm, and emotional tone, the delivery is calm and soothing, perfect for storytelling, the audio is crystal clear and sounds very close up."
)

# =============================================================================
# TEXT PROCESSING CONFIGURATION
# =============================================================================
# Regex pattern used to split long text into manageable chunks.
# Splits on Bengali Dari (।), Question marks (?), Exclamation marks (!), and Newlines.
SENTENCE_SPLIT_REGEX = r'([।?!\n])'

# Duration of silence (in seconds) inserted between text chunks for natural pacing.
CHUNK_SILENCE_DURATION = 0.2

# =============================================================================
# SERVER CONFIGURATION
# =============================================================================
# Host '0.0.0.0' allows external access (required for Docker/Cloud deployment).
HOST = "0.0.0.0"
PORT = 8000
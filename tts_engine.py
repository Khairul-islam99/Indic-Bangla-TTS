import torch
import soundfile as sf
import io
import logging
import numpy as np
import re
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
from tqdm import tqdm  # Import for progress visualization
import config  # Import centralized configuration

# Configure logging to standard output with timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class IndicTTSModel:
    """
    Wrapper class for the Indic Parler TTS model.
    Manages model initialization, text chunking, and audio synthesis with progress tracking.
    """

    def __init__(self):
        """
        Initializes the model and moves it to the appropriate device (GPU/CPU).
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing TTS Service on device: {self.device}...")
        
        try:
            # Load the model and tokenizers using configuration constants
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(config.MODEL_ID).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
            self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
            
            # Cache sampling rate
            self.sampling_rate = self.model.config.sampling_rate
            logger.info("Model and tokenizers loaded successfully.")
            
        except Exception as e:
            logger.critical(f"Failed to load model architecture. Error: {e}")
            raise e

    def split_text(self, text: str) -> list[str]:
        """
        Splits the input text into manageable chunks based on the regex pattern defined in config.
        This ensures long paragraphs are processed without losing context.

        Args:
            text (str): The raw input text.

        Returns:
            list[str]: A list of text segments.
        """
        # Split text while retaining delimiters (e.g., punctuation)
        parts = re.split(config.SENTENCE_SPLIT_REGEX, text)
        
        sentences = []
        current_sent = ""
        
        for part in parts:
            current_sent += part
            # Check if the part is a valid delimiter
            if part in "।?!\n":
                if current_sent.strip():
                    sentences.append(current_sent.strip())
                current_sent = ""
        
        # Append any remaining text
        if current_sent.strip():
            sentences.append(current_sent.strip())
            
        return sentences

    def generate_audio(self, text: str) -> io.BytesIO:
        """
        Synthesizes audio from text, displaying a progress bar in the console.

        Args:
            text (str): Input text.

        Returns:
            io.BytesIO: In-memory WAV file buffer.
        """
        try:
            chunks = self.split_text(text)
            logger.info(f"Processing request: {len(chunks)} chunks identified.")

            all_audio_arrays = []

            # Pre-compute description tokens (optimization)
            description_input_ids = self.description_tokenizer(
                config.VOICE_DESCRIPTION, return_tensors="pt"
            ).to(self.device)

            # Iterate over chunks with a progress bar (tqdm)
            # 'desc' sets the label, 'unit' sets the item name
            progress_bar = tqdm(chunks, desc="Synthesizing Audio", unit="chunk", colour="green")
            
            for chunk in progress_bar:
                if not chunk.strip(): 
                    continue
                
                # Tokenize the current text chunk
                prompt_input_ids = self.tokenizer(chunk, return_tensors="pt").to(self.device)

                # Generate audio tensor
                generation = self.model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask
                )

                # Post-process: Move to CPU, convert to numpy, squeeze dimensions
                audio_arr = generation.cpu().numpy().squeeze()
                all_audio_arrays.append(audio_arr)
                
                # Insert silence between chunks for natural pacing
                silence_samples = int(config.CHUNK_SILENCE_DURATION * self.sampling_rate)
                all_audio_arrays.append(np.zeros(silence_samples))

            if not all_audio_arrays:
                raise ValueError("No audio data generated.")

            # Concatenate all audio segments
            final_audio = np.concatenate(all_audio_arrays)

            # Write to buffer
            buffer = io.BytesIO()
            sf.write(buffer, final_audio, self.sampling_rate, format='WAV')
            buffer.seek(0)
            
            logger.info("Audio synthesis completed successfully.")
            return buffer

        except Exception as e:
            logger.error(f"Error during audio generation: {e}")
            raise e

# Instantiate the singleton engine
tts_engine = IndicTTSModel()
import json
import mimetypes
import os
import logging
from tqdm import tqdm
import google.generativeai as genai
from Functions.utils import wait_for_files_active, upload_to_gemini

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_audio_file_paths(directory: str) -> list[str]:
    """Gets audio file paths from a directory."""
    audio_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and mimetypes.guess_type(filepath)[0].startswith('audio/'):
            audio_files.append(filepath)
    return audio_files

def extract_json_from_audio(file_paths: list[str], _user_prompt: str, output_directory: str, model_name: str = "gemini-1.5-flash"):
    """Transcribes audio files and saves the output as JSON."""

    generation_config = {
        "temperature": 0.2,  # Reduced temperature for more focused responses
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    system_prompt = """
        You are a highly accurate and reliable transcription and analysis system. Your primary purpose is to analyze audio files, extract the spoken content, and provide a precise transcription of the conversation. Ensure that the transcription:
        1. Clearly identifies different speakers when possible.
        2. Accurately captures spoken words, including pauses, filler words, and nuances where appropriate.
        3. Correctly handles technical terms, proper nouns, and contextual details.
        4. Retains formatting for readability while maintaining fidelity to the original speech.
        
        You should aim to deliver results optimized for clarity and accuracy, even in cases of overlapping dialogue, background noise, or accents.
    """

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        system_instruction=system_prompt,
    )

    os.makedirs(output_directory, exist_ok=True)

    for file_path in tqdm(file_paths, desc="Processing audio files"):
        try:
            mime_type = mimetypes.guess_type(file_path)[0]
            if not mime_type.startswith('audio/'):
                logging.warning(f"Skipping non-audio file: {file_path}")
                continue

            logging.info(f"Uploading: {file_path}")
            uploaded_parts = upload_to_gemini(path=file_path, mime_type=mime_type)

            if not isinstance(uploaded_parts, list): # Ensure it's a list; handle if not
                uploaded_parts = [uploaded_parts]  # Wrap in a list if single File object.

            wait_for_files_active(uploaded_parts)

            chat_session = model.start_chat(history=[{"role": "user", "parts": uploaded_parts}])
            response = chat_session.send_message(_user_prompt)

            json_path = os.path.join(output_directory, os.path.splitext(os.path.basename(file_path))[0] + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json.loads(response.text), f, indent=4) # Directly use to_json()

            logging.info(f"Saved transcription to: {json_path}")

        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")



if __name__ == "__main__":
    audio_paths = get_audio_file_paths("AudioData")
    user_prompt = """
        Please analyze the provided audio file and transcribe the conversation with high accuracy.
        - Identify and label different speakers if discernible (e.g., Speaker 1, Speaker 2).
        - Capture all spoken words, including filler words (e.g., um, ah) and hesitations.
        - Ensure correct spelling of technical terms and proper nouns.
        - If overlapping speech occurs, indicate it clearly.
        - Provide timestamps at regular intervals or speaker changes for easy reference.
    """
    extract_json_from_audio(audio_paths, user_prompt, "TextOutput")
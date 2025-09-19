import falcon
import requests
from dotenv import load_dotenv
assert load_dotenv("env.env"), "Environment File not Found."
import os
import re
import json
from time import sleep
from base64 import b64decode
from dataclasses import dataclass
from dataclasses_json import dataclass_json

STREAM_THRESHOLD = 1024 * 10
MAX_UPLOAD_SIZE = 1024 * 1024 * 100

cartesia_voices = {'laidback woman', 'polite man', 'storyteller lady', 'friendly sidekick'}
cartesia_formats = {'mp3', 'wav'}
cartesia_languages = {'en', 'de', 'fr', 'es', 'hi', 'it', 'ja', 'ko', 'nl', 'pl', 'pt', 'ru', 'sv', 'tr', 'zh'}
cartesia_language_id_dict = {0: 'en', 1: 'hi', 21: 'es', 22: 'it', 23: 'de'}
language_id_dict = {0: "English", 1: "Hindi", 21: "Spanish", 22: "Italian", 23: "German"}
language_code_to_id = {"en": 0, "hi": 1, "es": 21, "it": 22, "de": 23}

LANGUAGE_DETECT_PROMPT = """Detect the language(s) in the text. Return STRICT JSON matching this schema:
{
  "language_code": "string",           // ISO 639-1 if available, else 639-3
  "language_name": "string",
  "script": "string",                  // e.g., Latin, Devanagari
  "confidence": 0.0,                   // 0.0–1.0 reflecting your certainty
  "is_code_mixed": true,
  "segments": [
    {"text": "string", "language_code": "string"}
  ]
}

Rules:
- If code-mixed, choose the dominant language for "language_code".
- If text is too short/ambiguous (e.g., "ok"), set confidence ≤ 0.6.
- Use "und" for undetermined.
Return JSON ONLY—no commentary."""

LANGUAGE_TRANSLATE_PROMPT = """You are a professional translator. Preserve meaning, tone, numbers, and URLs. Keep markup and placeholders unchanged.

Translate from {SOURCE_LANG} to {TARGET_LANG}.
Constraints:
- If the source contains markup (HTML/Markdown), preserve structure.
- Maintain units; convert only if explicitly asked (default: do not).
- Output: ONLY the translated text (no explanations)."""

JSON_FIX_PROMPT = """You are a JSON syntax corrector.

Input:
1. A possibly malformed JSON string.
2. A parser error message that hints at where the problem is.

Task:
- Output valid JSON that preserves the original structure and data as much as possible.
- Fix only syntax issues (e.g. missing commas, unescaped quotes, mismatched brackets, trailing commas).
- Do not invent new keys or values; only repair what is broken.
- Ensure the output parses cleanly with a standard JSON parser.

Return ONLY the corrected JSON, nothing else."""

def strip_json_fencing(text: str) -> str:
    match = re.search(r"```(?:json)?(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip("`").lstrip("json").strip()

@dataclass_json
@dataclass
class GData:
    text: str
    voice_name: str="laidback woman"
    audio_language: str='en'
    audio_format: str="wav"

@dataclass_json
@dataclass
class SData:
    audio_base64: str
    audio_language: str='en'

@dataclass_json
@dataclass
class TData:
    text: str
    from_id: int
    to_id: int

@dataclass_json
@dataclass
class DData:
    text: str

def generate_audio(text: str, voice_name: str, language: str, audio_format: str, retries: int=0, max_retries: int=3) -> bytes:
    headers = {"Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ['togetherai_api_key']}"}
    payload = {"model": "cartesia/sonic-2", "input": text, "voice": voice_name,
    'language': language}
    try:
        r = requests.post(os.environ['togetherai_api_audio_endpoint'], json=payload, headers=headers)
        r.raise_for_status()
    except Exception as exp:
        if retries >= max_retries:
            raise RuntimeError("Max retries received.")
        sleep(1)
        return generate_audio(text, voice_name, language, audio_format, retries+1)
    else:
        return r.content

def recognize_audio(audio_bytes: bytes, language: str='en', retries: int=0, max_retries: int=3) -> str:
    headers = {"Authorization": f"Bearer {os.environ['togetherai_api_key']}"}
    files = {"file": audio_bytes}
    data = {
    "model": "openai/whisper-large-v3",
    "language": language
    }
    try:
        r = requests.post(os.environ['togetherai_api_recognize_endpoint'],
        files=files, data=data, headers=headers)
        r.raise_for_status()
    except Exception as exp:
        if retries >= max_retries:
            raise RuntimeError("Max retries received.")
        sleep(1)
        return recognize_audio(audio_bytes, language, retries+1)
    else:
        return r.json()['text']

def call_llm(system_prompt: str, broken: str, error_msg: str) -> str:
    headers = {"Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ['togetherai_api_key']}"}
    payload = {"messages": [{"role": "system", "content": system_prompt},
    {"role": "user", "content": broken}], "model": "openai/gpt-oss-120b"}
    r = requests.post(os.environ['togetherai_api_endpoint'], headers=headers, json=payload)
    return r.json()["choices"][0]["message"]["content"]

def fix_json(broken: str, error_msg: str, max_retries: int = 3) -> str:
    """Fix JSON string until it parses."""
    attempt = broken
    for _ in range(max_retries):
        try:
            json.loads(attempt)
            return attempt  # already valid
        except Exception as e:
            attempt = call_llm(JSON_FIX_PROMPT, attempt, str(e))
    # final check
    attempt = strip_json_fencing(attempt)
    json.loads(attempt)
    return attempt

def llm_language(system_prompt: str, prompt: str, retries: int=0, max_retries: int=3):
    headers = {"Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ['togetherai_api_key']}"}
    payload = {"messages": [{"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}], "model": "openai/gpt-oss-20b"}
    try:
        r = requests.post(os.environ['togetherai_api_endpoint'], json=payload, headers=headers)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as exp:
        if retries >= max_retries:
            raise RuntimeError("Max retries reached.")
        sleep(1)
        return llm_language(system_prompt, prompt, retries+1)


class RecognizeResource:
    def on_post(self, req, resp):
        api_key = req.get_header('api-key')
        if api_key != os.environ['api_key']:
            resp.status = falcon.HTTP_401
            resp.text = "Authorization Failed"
            return

        try:
            data = SData.from_dict(req.media)
        except Exception as exp:
            resp.status = falcon.HTTP_422
            resp.text = str(exp)
            return
        try:
            audio_bytes = b64decode(data.audio_base64.encode())
        except Exception as exp:
            resp.status = falcon.HTTP_424
            resp.text = "Failed to read audio data"
            return
        language = data.audio_language if data.audio_language in cartesia_languages else 'en'
        resp.media = {"text": recognize_audio(audio_bytes, language=language)}
        resp.status = falcon.HTTP_200
        return

class GenerateResource:
    def on_post(self, req, resp):
        api_key = req.get_header('api-key')
        if api_key != os.environ['api_key']:
            resp.status = falcon.HTTP_401
            resp.text = "Authorization Failed"
            return
        try:
            data = GData.from_dict(req.media)
        except Exception as exp:
            resp.status = falcon.HTTP_422
            resp.text = str(exp)
            return
        audio_language = data.audio_language if data.audio_language in cartesia_languages else 'en'
        voice_name = data.voice_name if data.voice_name in cartesia_voices else 'laidback woman'
        audio_format = data.audio_format if data.audio_format in cartesia_formats else 'wav'
        resp.data = generate_audio(data.text, voice_name, audio_language, audio_format)
        resp.status = falcon.HTTP_200
        return

class TranslateResource:
    def on_post(self, req, resp):
        api_key = req.get_header('api-key')
        if api_key != os.environ['api_key']:
            resp.status = falcon.HTTP_401
            resp.text = "Authorization Failed"
            return
        try:
            data = TData.from_dict(req.media)
        except Exception as exp:
            resp.status = falcon.HTTP_422
            resp.text = str(exp)
            return
        system_prompt = LANGUAGE_TRANSLATE_PROMPT.format(SOURCE_LANG=language_id_dict.get(data.from_id, 'UNKNOWN'),
        TARGET_LANG=language_id_dict.get(data.to_id, 'English'))
        translated_text = llm_language(system_prompt, data.text)
        resp.status = falcon.HTTP_200
        resp.media = {'translated_text': translated_text}
        return

class DetectResource:
    def on_post(self, req, resp):
        api_key = req.get_header('api-key')
        if api_key != os.environ['api_key']:
            resp.status = falcon.HTTP_401
            resp.text = "Authorization Failed"
            return
        try:
            data = DData.from_dict(req.media)
        except Exception as exp:
            resp.status = falcon.HTTP_422
            resp.text = str(exp)
            return
        system_prompt = LANGUAGE_DETECT_PROMPT
        response_json_string = strip_json_fencing(llm_language(system_prompt, data.text))
        try:
            response = json.loads(response_json_string)
        except json.JSONDecodeError as exp:
            response = json.loads(fix_json(response_json_string, str(exp)))
        finally:
            resp.media = {"language_id": language_code_to_id.get(response['language_code'], None),
            'language_name': response['language_name']}
            resp.status = falcon.HTTP_200
            return


app = falcon.App()
handler = falcon.media.MultipartFormHandler()
handler.parse_options.max_body_part_buffer_size = MAX_UPLOAD_SIZE
app.req_options.media_handlers[falcon.MEDIA_MULTIPART] = handler

app.add_route("/recognize", RecognizeResource())
app.add_route("/generate", GenerateResource())
app.add_route("/translate", TranslateResource())
app.add_route("/detect", DetectResource())

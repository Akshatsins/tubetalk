import yt_dlp
import os
from groq import Groq

def download_audio(youtube_url: str, output_path: str) -> str:
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace('.mp3', ''),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    
    # yt-dlp appends .mp3 automatically
    final_path = output_path if output_path.endswith('.mp3') else output_path + '.mp3'
    return final_path


def transcribe_audio(audio_path: str, api_key: str) -> list:
    """Transcribe using Groq's Whisper API. Returns list of segments with text and start time."""
    client = Groq(api_key=api_key)
    
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    
    segments = []
    if hasattr(transcription, 'segments') and transcription.segments:
        for seg in transcription.segments:
            segments.append({
                'text': seg.text.strip(),
                'start': seg.start,
            })
    else:
        # Fallback: treat whole transcript as one segment
        segments = [{'text': transcription.text, 'start': 0.0}]
    
    return segments


def delete_file(path: str):
    if path and os.path.exists(path):
        os.remove(path)

from youtube_transcript_api import YouTubeTranscriptApi
import re

def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def get_transcript(youtube_url: str) -> list:
    """
    Fetch transcript from YouTube using the Transcript API.
    Returns a list of segments with 'text' and 'start' keys.
    """
    video_id = extract_video_id(youtube_url)

    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Try manually created captions first, then auto-generated
    try:
        transcript = transcript_list.find_manually_created_transcript(
            ['en', 'en-US', 'en-GB']
        )
    except Exception:
        try:
            transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
        except Exception:
            # Fall back to whatever language is available and translate
            transcript = next(iter(transcript_list))
            transcript = transcript.translate('en')

    raw = transcript.fetch()

    segments = [
        {'text': entry['text'].strip(), 'start': entry['start']}
        for entry in raw
        if entry['text'].strip()
    ]
    return segments


# Keep these for backward compatibility (app.py calls delete_file)
def download_audio(youtube_url: str, output_path: str) -> str:
    """Not used anymore — transcript is fetched directly."""
    raise NotImplementedError("Use get_transcript() instead.")

def delete_file(path: str):
    """No-op since we no longer download files."""
    pass

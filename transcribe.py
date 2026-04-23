from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re
import os
 
 
def extract_video_id(url: str) -> str:
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
 
 
def _build_ytt():
    """
    Build a YouTubeTranscriptApi instance.
    If a proxy URL is provided via the HTTPS_PROXY env var or
    Streamlit secrets (set HTTPS_PROXY in your secrets.toml), it is
    used to route requests around YouTube's server-IP block.
 
    secrets.toml example:
        HTTPS_PROXY = "https://user:pass@proxyhost:port"
    """
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
 
    # Also try to pull from Streamlit secrets without hard-importing streamlit
    if not proxy_url:
        try:
            import streamlit as st
            proxy_url = st.secrets.get("HTTPS_PROXY", "")
        except Exception:
            pass
 
    if proxy_url:
        proxies = {"http": proxy_url, "https": proxy_url}
        return YouTubeTranscriptApi(proxies=proxies)
 
    return YouTubeTranscriptApi()
 
 
def get_transcript(youtube_url: str) -> list:
    video_id = extract_video_id(youtube_url)
 
    try:
        ytt = _build_ytt()
    except TypeError:
        # Older versions of youtube-transcript-api don't accept proxies
        # in the constructor; fall back to the module-level call with
        # the requests_kwargs approach.
        ytt = None
 
    try:
        if ytt is not None:
            transcript_list = ytt.list(video_id)
        else:
            # Fallback: legacy API (< 0.6.x) — pass proxies via requests_kwargs
            proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
            kwargs = {}
            if proxy_url:
                kwargs["proxies"] = {"http": proxy_url, "https": proxy_url}
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id, **kwargs)
 
    except Exception as e:
        err = str(e).lower()
        if "ip" in err or "block" in err or "429" in err or "forbidden" in err or "could not retrieve" in err:
            raise RuntimeError(
                "YouTube is blocking transcript requests from this server's IP address. "
                "To fix this, add a proxy URL to your Streamlit secrets:\n\n"
                "  HTTPS_PROXY = \"https://user:pass@proxyhost:port\"\n\n"
                "Free options: webshare.io (10 free proxies) or use a residential proxy service."
            ) from e
        raise
 
    # --- Transcript selection (manual → generated → translated) ---
    try:
        transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
    except Exception:
        try:
            transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
        except Exception:
            try:
                transcript = next(iter(transcript_list)).translate('en')
            except Exception as e:
                raise RuntimeError(
                    "No usable transcript found for this video. "
                    "The video may not have captions enabled."
                ) from e
 
    raw = transcript.fetch()
    segments = []
    for entry in raw:
        if hasattr(entry, 'text'):
            text, start = entry.text.strip(), entry.start
        else:
            text, start = entry['text'].strip(), entry['start']
        if text:
            segments.append({'text': text, 'start': start})
 
    return segments
 
 
def delete_file(path: str):
    pass
 

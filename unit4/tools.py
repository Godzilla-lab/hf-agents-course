"""
The toolbox. Every function here is something the agent cannot do on its own.

Why do these exist? The agent writes Python, but that Python runs in a sandbox
that forbids reading files, spawning processes, or hitting the network. So any
job that needs the real world (a file on disk, a web page, an audio clip) is
wrapped in a @tool function. The agent calls the tool by name; the tool runs as
normal Python outside the sandbox and hands back a string.

A @tool needs three things or smolagents refuses it at import time:
  1. type hints on every argument and the return value
  2. a docstring with an "Args:" section describing each argument
  3. a return value that is a string (or something printable)
The docstring is not decoration: it is literally what the model reads to decide
when to use the tool. Write it for the model.
"""

import base64
import os
import re
import subprocess
import sys
import time

import requests
from dotenv import load_dotenv
from markdownify import markdownify
from smolagents import DuckDuckGoSearchTool, tool

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

API_BASE = "https://agents-course-unit4-scoring.hf.space"
FILES_DIR = os.path.join(HERE, "files")

# The "senses" models: audio, images, video. Gemini does all three natively.
# The free tier allows only ~20 requests per DAY per model, but every model has
# its own bucket, so we walk down this list when one runs dry.
MEDIA_MODELS = [
    m.strip().split("/", 1)[-1]
    for m in os.environ.get(
        "MEDIA_MODELS",
        "gemini/gemini-3.8-flash,gemini/gemini-3.6-flash,gemini/gemini-3.5-flash,"
        "gemini/gemini-3.5-flash-lite,gemini/gemini-3.1-flash-lite,gemini/gemini-3-flash-preview,"
        "gemini/gemini-flash-latest,gemini/gemini-flash-lite-latest",
    ).split(",")
    if m.strip()
]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
EXHAUSTED_MODELS: set[str] = set()  # models that returned a daily-quota error this process

MIME_TYPES = {
    ".mp3": "audio/mp3", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".pdf": "application/pdf",
}


# ---------------------------------------------------------------- helpers (not tools)

def fetch_task_file(task_id: str, file_name: str) -> str:
    """
    Download a question's attachment once, cache it in files/, return the absolute path.

    Tries the course API first. That endpoint has been known to 404, so as a
    fallback we pull the identical file from the official GAIA dataset on the
    Hugging Face Hub (gated: your HF token must have accepted its terms).
    """
    os.makedirs(FILES_DIR, exist_ok=True)
    path = os.path.join(FILES_DIR, file_name)
    if os.path.exists(path):
        return path

    resp = requests.get(f"{API_BASE}/files/{task_id}", timeout=60)
    if resp.status_code == 200:
        with open(path, "wb") as f:
            f.write(resp.content)
        return path

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        filename=f"2023/validation/{file_name}",
        token=os.environ.get("HF_TOKEN"),  # None means: use the token from `hf auth login`
    )
    with open(downloaded, "rb") as src, open(path, "wb") as dst:
        dst.write(src.read())
    return path


def _ask_gemini(parts: list[dict], attempts_per_model: int = 3) -> str:
    """
    One raw call to the Gemini REST API. No SDK, no adapter: a URL, a JSON body,
    a JSON reply. `parts` is a list of content pieces (text, inline bytes, or a
    file URL) that Gemini reads together as one message.

    Walks the MEDIA_MODELS list: a model that is out of daily quota (or not
    available to this key) is skipped for the rest of the process; a model that
    is merely overloaded (503) is retried a few times before moving on.
    """
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return "ERROR: GEMINI_API_KEY is not set, media tools are unavailable."
    body = {"contents": [{"role": "user", "parts": parts}]}
    last_error = "no media model available"
    for model in MEDIA_MODELS:
        if model in EXHAUSTED_MODELS:
            continue
        url = GEMINI_URL.format(model=model)
        delay = 8
        for attempt in range(1, attempts_per_model + 1):
            try:
                resp = requests.post(url, params={"key": key}, json=body, timeout=600)
            except requests.RequestException as e:
                last_error = f"{model}: {e}"
                time.sleep(delay)
                continue
            if resp.status_code == 200:
                data = resp.json()
                try:
                    return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"]).strip()
                except (KeyError, IndexError):
                    return f"ERROR: unexpected Gemini reply from {model}: {str(data)[:500]}"
            text = resp.text
            last_error = f"{model}: HTTP {resp.status_code}: {text[:300]}"
            daily_cap = resp.status_code == 429 and ("free_tier" in text or "per_day" in text.lower() or "PerDay" in text)
            if resp.status_code == 404 or daily_cap:
                EXHAUSTED_MODELS.add(model)  # useless for the rest of the run, move on
                break
            if resp.status_code in (429, 500, 503) and attempt < attempts_per_model:
                time.sleep(delay)
                delay *= 2
                continue
            break  # other error: try the next model
    return f"ERROR: all media models failed. Last error: {last_error}"


def _inline_part(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    mime = MIME_TYPES.get(ext)
    if not mime:
        raise ValueError(f"unsupported media type: {ext}")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"inline_data": {"mime_type": mime, "data": data}}


# ---------------------------------------------------------------- file tools

@tool
def read_text_file(path: str) -> str:
    """
    Read a text file (for example .py, .txt, .csv, .md, .json) and return its contents.
    Use this to LOOK at a script before you run it.

    Args:
        path: Absolute path to the file on disk.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text[:20000] + ("\n...[truncated]" if len(text) > 20000 else "")


@tool
def run_python_file(path: str) -> str:
    """
    Execute a Python script in a separate process and return everything it printed
    (stdout, then stderr) plus its exit code. Times out after 120 seconds.

    Args:
        path: Absolute path to the .py file to run.
    """
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "ERROR: script timed out after 120 seconds"
    out = proc.stdout[-3000:]
    err = proc.stderr[-1500:]
    return f"exit code: {proc.returncode}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"


@tool
def read_excel_file(path: str) -> str:
    """
    Read every sheet of an Excel workbook (.xlsx/.xls) and return them as CSV text,
    one block per sheet, with the sheet name and its row/column counts. Then do the
    maths yourself: `import pandas as pd, io; df = pd.read_csv(io.StringIO(csv_text))`.

    Args:
        path: Absolute path to the Excel file.
    """
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None)
    chunks = []
    for name, df in sheets.items():
        chunks.append(f"### sheet: {name}  ({len(df)} rows x {len(df.columns)} columns)\n{df.to_csv(index=False)}")
    text = "\n".join(chunks)
    return text[:30000] + ("\n...[truncated]" if len(text) > 30000 else "")


# ---------------------------------------------------------------- media tools

@tool
def transcribe_audio(path: str) -> str:
    """
    Transcribe an audio file (.mp3, .wav, .m4a) to text, word for word.

    Args:
        path: Absolute path to the audio file.
    """
    prompt = ("Transcribe this audio verbatim. Output only the transcript, no commentary. "
              "Write numbers as digits.")
    return _ask_gemini([_inline_part(path), {"text": prompt}])


@tool
def analyze_image(path: str, question: str) -> str:
    """
    Look at an image (.png, .jpg, .webp) and answer a question about it. Be specific
    in the question: say exactly what you want listed or described.

    Args:
        path: Absolute path to the image file.
        question: What you want to know about the image.
    """
    return _ask_grok_vision([path], question)


VIDEO_DIR = os.path.join(FILES_DIR, "video")
FRAME_EVERY_SECONDS = 2
MAX_FRAMES = 96
FRAMES_PER_BATCH = 12


def _download_video(url: str) -> dict:
    """yt-dlp: small mp4 video stream, English captions if any, and the audio track."""
    import glob

    import yt_dlp

    os.makedirs(VIDEO_DIR, exist_ok=True)
    common = {"outtmpl": os.path.join(VIDEO_DIR, "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True, "noprogress": True}
    with yt_dlp.YoutubeDL({**common, "format": "bv*[height<=480][ext=mp4]/bv*[height<=480]/bv*",
                           "writesubtitles": True, "writeautomaticsub": True,
                           "subtitleslangs": ["en", "en-orig", "en-US"], "subtitlesformat": "vtt"}) as ydl:
        info = ydl.extract_info(url, download=True)
    vid = info["id"]
    video = next(iter(sorted(glob.glob(os.path.join(VIDEO_DIR, f"{vid}.mp4")) + glob.glob(os.path.join(VIDEO_DIR, f"{vid}.webm")))), None)
    subs = sorted(glob.glob(os.path.join(VIDEO_DIR, f"{vid}*.vtt")))
    audio = os.path.join(VIDEO_DIR, f"{vid}.m4a")
    if not os.path.exists(audio):
        try:
            with yt_dlp.YoutubeDL({**common, "format": "ba[ext=m4a]/ba"}) as ydl:
                ydl.download([url])
        except Exception:
            audio = None
    return {"id": vid, "title": info.get("title"), "duration": info.get("duration"),
            "video": video, "subs": subs[0] if subs else None, "audio": audio if audio and os.path.exists(audio) else None}


def _vtt_to_text(path: str) -> str:
    """Flatten a .vtt caption file into '[mm:ss] words' lines without duplicates."""
    lines, seen, stamp = [], set(), ""
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if "-->" in raw:
                stamp = raw.split("-->")[0].strip()[3:8]
                continue
            if not raw or raw.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")):
                continue
            text = re.sub(r"<[^>]+>", "", raw).strip()
            if text and text not in seen:
                seen.add(text)
                lines.append(f"[{stamp}] {text}")
    return "\n".join(lines)


def _sample_frames(video_path: str) -> list[tuple[float, bytes]]:
    """Grab one JPEG every FRAME_EVERY_SECONDS, capped at MAX_FRAMES, shrunk to 640px wide."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    step = max(FRAME_EVERY_SECONDS, duration / MAX_FRAMES)
    frames, t = [], 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            frames.append((t, buf.tobytes()))
        t += step
    cap.release()
    return frames


@tool
def watch_youtube_video(url: str, question: str) -> str:
    """
    Watch a YouTube video and answer a question about it. The video is downloaded,
    its captions are read, frames are sampled every couple of seconds and shown to a
    vision model in batches, and the batch notes are combined into one answer.
    Ask precisely: for example "What is the highest number of bird species visible at
    the same moment?" or "Quote the exact words Teal'c says in reply to 'Isn't that hot?'".

    Args:
        url: The full YouTube URL, for example https://www.youtube.com/watch?v=abc123
        question: What you want to know about the video.
    """
    try:
        meta = _download_video(url)
    except Exception as e:
        return f"ERROR: could not download video: {e}"

    transcript = _vtt_to_text(meta["subs"]) if meta["subs"] else ""
    if not transcript and meta["audio"]:
        transcript = _ask_gemini([_inline_part_bytes(meta["audio"], "audio/mp4"),
                                  {"text": "Transcribe this audio verbatim with [mm:ss] timestamps every few seconds."}])
        if transcript.startswith("ERROR"):
            transcript = ""
    header = f"Video: {meta['title']!r}, duration {meta['duration']} s"

    notes = []
    if meta["video"]:
        frames = _sample_frames(meta["video"])
        for i in range(0, len(frames), FRAMES_PER_BATCH):
            batch = frames[i:i + FRAMES_PER_BATCH]
            stamps = ", ".join(f"{t:.0f}s" for t, _ in batch)
            prompt = (f"{header}. These {len(batch)} frames were captured at {stamps}, in order. "
                      f"For the question below, describe precisely what each frame shows that is relevant, "
                      f"and note the frame timestamp. Question: {question}")
            notes.append(f"--- frames {stamps} ---\n" + _ask_grok_vision([("image/jpeg", b) for _, b in batch], prompt))
    else:
        notes.append("(no video stream could be downloaded, only captions)")

    synthesis_prompt = (
        f"{header}\n\nQUESTION: {question}\n\n"
        f"CAPTIONS/TRANSCRIPT:\n{transcript[:8000] or '(none)'}\n\n"
        f"FRAME NOTES from a vision model:\n{chr(10).join(notes)[:24000]}\n\n"
        "Using the captions and frame notes together, answer the question as precisely as "
        "possible. State the answer first, then one line of justification with timestamps."
    )
    answer = _ask_grok_vision([], synthesis_prompt)
    return f"{answer}\n\n[transcript excerpt]\n{transcript[:1500] or '(none)'}"


def _inline_part_bytes(path: str, mime: str) -> dict:
    with open(path, "rb") as f:
        return {"inline_data": {"mime_type": mime, "data": base64.b64encode(f.read()).decode()}}


# ---------------------------------------------------------------- web tools

# Wikipedia and many sites refuse requests with no browser-like User-Agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 gaia-agent/1.0"}
PAGE_CACHE: dict[str, str] = {}  # url -> markdown text, so find_in_page never refetches
PAGE_CHARS = 20000


def _html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = markdownify(html, heading_style="ATX", strip=["img", "svg"])
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_page(url: str) -> str:
    if url in PAGE_CACHE:
        return PAGE_CACHE[url]
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "")
    text = _html_to_text(resp.text) if "html" in ctype else resp.text
    PAGE_CACHE[url] = text
    return text


web_search = DuckDuckGoSearchTool(max_results=8)


@tool
def visit_webpage(url: str, start: int = 0) -> str:
    """
    Fetch a web page and return its text as markdown, 20000 characters at a time.
    Call again with a larger `start` to read further, or use find_in_page to jump
    straight to the part you need.

    Args:
        url: Full URL, including https://
        start: Character offset to start from (0 for the top of the page).
    """
    try:
        text = _get_page(url)
    except Exception as e:
        return f"ERROR fetching {url}: {e}"
    chunk = text[start:start + PAGE_CHARS]
    tail = f"\n\n[showing chars {start}-{start + len(chunk)} of {len(text)}]"
    return chunk + tail


@tool
def find_in_page(url: str, query: str, context: int = 400) -> str:
    """
    Search inside a web page for a word or phrase (case-insensitive) and return
    each match with surrounding text. Much cheaper than reading the whole page.

    Args:
        url: Full URL of the page.
        query: Word or phrase to look for.
        context: How many characters to show on each side of every match.
    """
    try:
        text = _get_page(url)
    except Exception as e:
        return f"ERROR fetching {url}: {e}"
    hits = [m.start() for m in re.finditer(re.escape(query), text, flags=re.I)]
    if not hits:
        return f"'{query}' not found in {url} ({len(text)} chars)"
    out = []
    last_end = -1
    for pos in hits[:12]:
        a, b = max(0, pos - context), min(len(text), pos + len(query) + context)
        if a < last_end:
            continue
        out.append(f"--- match at char {pos} ---\n{text[a:b]}")
        last_end = b
    return f"{len(hits)} matches for '{query}' in {url}\n\n" + "\n\n".join(out)


WIKI_API = "https://en.wikipedia.org/w/api.php"


@tool
def wikipedia_page(title: str, as_of_date: str = "") -> str:
    """
    Fetch an English Wikipedia article as markdown, optionally as it looked on a
    given date (useful when a question says "the 2022 version of Wikipedia").
    Returns the first 12000 characters plus the URL, which you can then pass to
    find_in_page or visit_webpage(url, start=...) to read more.

    Args:
        title: Article title, for example "Mercedes Sosa".
        as_of_date: Optional YYYY-MM-DD. The latest revision on or before this date is used.
    """
    params = {"action": "parse", "page": title, "prop": "text", "format": "json", "redirects": 1}
    if as_of_date:
        rev = requests.get(WIKI_API, headers=HEADERS, timeout=30, params={
            "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
            "rvstart": f"{as_of_date}T23:59:59Z", "rvdir": "older", "format": "json", "redirects": 1,
        }).json()
        pages = rev.get("query", {}).get("pages", {})
        revisions = next(iter(pages.values()), {}).get("revisions", [])
        if not revisions:
            return f"No revision of '{title}' found on or before {as_of_date}"
        params = {"action": "parse", "oldid": revisions[0]["revid"], "prop": "text", "format": "json"}
    data = requests.get(WIKI_API, headers=HEADERS, timeout=30, params=params).json()
    if "error" in data:
        return f"Wikipedia error: {data['error'].get('info')}"
    html = data["parse"]["text"]["*"]
    url = f"https://en.wikipedia.org/wiki/{data['parse']['title'].replace(' ', '_')}"
    if as_of_date:
        url += f"?oldid={params['oldid']}"
    text = _html_to_text(html)
    PAGE_CACHE[url] = text
    return f"URL: {url}\n\n{text[:PAGE_CHARS]}\n\n[showing chars 0-{min(PAGE_CHARS, len(text))} of {len(text)}]"


# ---------------------------------------------------------------- chess tools

BOARD_PROMPT = """This image is a chess diagram. Read it very carefully, square by square.
First state which colour is at the bottom of the board (look at the rank/file labels
if present; if white is at the bottom, a1 is the bottom-left corner; if black is at the
bottom, h8 is the bottom-left corner).
Then output ONLY a JSON object of the form
{"bottom": "white" or "black", "pieces": {"e1": "K", "e8": "k", ...}}
using FEN letters: uppercase for white (K Q R B N P), lowercase for black (k q r b n p).
List every piece on the board. Do not include empty squares. Output nothing else."""


def _ask_grok_vision(images: list, prompt: str) -> str:
    """
    The brain model looking at pictures. `images` is a list of file paths or of
    (mime_type, raw_bytes) tuples. Grok reads text and images, not audio or video.
    """
    import litellm

    content = [{"type": "text", "text": prompt}]
    for img in images:
        if isinstance(img, tuple):
            mime, raw = img
        else:
            mime = MIME_TYPES.get(os.path.splitext(img)[1].lower(), "image/png")
            with open(img, "rb") as f:
                raw = f.read()
        data = base64.b64encode(raw).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
    last = None
    for attempt in range(4):
        try:
            resp = litellm.completion(
                model=os.environ.get("AGENT_MODEL", "xai/grok-4.6"),
                api_key=os.environ.get("XAI_API_KEY"),
                messages=[{"role": "user", "content": content}],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last = e
            time.sleep(10 * (attempt + 1))
    return f"ERROR: vision call failed: {last}"


def _pieces_to_board(reply: str):
    """Turn a vision model's JSON piece list into a python-chess Board (no side/castling info)."""
    import json as _json

    import chess

    m = re.search(r"\{.*\}", reply, flags=re.S)
    if not m:
        raise ValueError("no JSON in reply")
    data = _json.loads(m.group(0))
    board = chess.Board(None)
    for sq, piece in data["pieces"].items():
        board.set_piece_at(chess.parse_square(sq.lower()), chess.Piece.from_symbol(piece))
    return board, data.get("bottom", "?")


@tool
def read_chess_board(path: str) -> str:
    """
    Read a chess diagram image and return the position as a FEN piece placement plus
    an ASCII board, from two independent vision models. If the two readings differ,
    the differing squares are listed so you can look again with analyze_image and
    decide. Use the result with chess_best_move.

    Args:
        path: Absolute path to the image of the chess board.
    """
    import chess

    readings = {}
    for name, fn in (("grok", lambda: _ask_grok_vision([path], BOARD_PROMPT)),
                     ("gemini", lambda: _ask_gemini([_inline_part(path), {"text": BOARD_PROMPT}]))):
        try:
            board, bottom = _pieces_to_board(fn())
            readings[name] = (board, bottom)
        except Exception as e:
            readings[name] = (None, f"failed: {e}")

    out = []
    for name, (board, bottom) in readings.items():
        if board is None:
            out.append(f"[{name}] {bottom}")
            continue
        out.append(f"[{name}] bottom={bottom}  placement={board.board_fen()}\n{board}")
    boards = [b for b, _ in readings.values() if b is not None]
    if len(boards) == 2:
        diffs = [chess.square_name(sq) for sq in chess.SQUARES if boards[0].piece_at(sq) != boards[1].piece_at(sq)]
        out.append("Both models AGREE on every square." if not diffs else f"Models DISAGREE on squares: {', '.join(diffs)}")
    out.append("To build a full FEN for chess_best_move: '<placement> <b or w> - - 0 1' "
               "(use 'b' if black is to move; adjust castling rights only if kings and rooks are on their home squares).")
    return "\n\n".join(out)


def _find_stockfish() -> str | None:
    import shutil

    for candidate in (shutil.which("stockfish"), "/usr/games/stockfish", "/usr/bin/stockfish",
                      os.path.join(HERE, "bin", "stockfish")):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


@tool
def chess_best_move(fen: str, depth: int = 22) -> str:
    """
    Ask the Stockfish chess engine for the best move in a position. Returns the move in
    standard algebraic notation (for example Rd5, Qxf7+, Nf3), the evaluation, and the
    expected continuation. Also lists all legal moves so you can double-check notation.

    Args:
        fen: The full FEN string, for example "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".
        depth: Search depth (higher is stronger but slower; 22 takes a few seconds).
    """
    import chess
    import chess.engine

    try:
        board = chess.Board(fen)
    except ValueError as e:
        return f"ERROR: invalid FEN: {e}"
    if not board.is_valid():
        return f"ERROR: position is not legal: {board.status()!r}. Check the piece placement."
    exe = _find_stockfish()
    if not exe:
        return "ERROR: Stockfish engine not found on this machine."
    with chess.engine.SimpleEngine.popen_uci(exe) as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=3)
    lines = []
    for i, cand in enumerate(info, start=1):
        pv = cand.get("pv", [])
        san = board.variation_san(pv[:8]) if pv else "?"
        score = cand["score"].pov(board.turn)
        lines.append(f"{i}. {board.san(pv[0]) if pv else '?'}  eval={score}  line: {san}")
    legal = ", ".join(board.san(m) for m in board.legal_moves)
    return (f"Side to move: {'black' if board.turn == chess.BLACK else 'white'}\n"
            + "\n".join(lines) + f"\n\nLegal moves: {legal}")


ALL_TOOLS = [
    web_search, visit_webpage, find_in_page, wikipedia_page,
    read_text_file, run_python_file, read_excel_file,
    transcribe_audio, analyze_image, watch_youtube_video,
    read_chess_board, chess_best_move,
]


if __name__ == "__main__":
    # Zero-LLM check: does each tool import and describe itself correctly?
    for t in ALL_TOOLS:
        print(f"{t.name:22} args={list(t.inputs)}")

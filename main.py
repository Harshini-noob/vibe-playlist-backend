from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, json, sqlite3
from groq import Groq
from dotenv import load_dotenv
import requests as req_lib
import shutil

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── DATABASE ─────────────────────────────────────────────────

# ─── DATABASE ─────────────────────────────────────────────────

DB_PATH = "feedback.db"

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_title TEXT NOT NULL,
            track_artist TEXT NOT NULL,
            mood TEXT NOT NULL,
            language TEXT,
            rating INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS track_scores (
            track_title TEXT NOT NULL,
            track_artist TEXT NOT NULL,
            mood TEXT NOT NULL,
            thumbs_up INTEGER DEFAULT 0,
            thumbs_down INTEGER DEFAULT 0,
            score REAL DEFAULT 0.0,
            PRIMARY KEY (track_title, track_artist, mood)
        )
    """)
    conn.commit()
    return conn

# single global connection — no locking issues
db = init_db()

def update_track_score(title: str, artist: str, mood: str, rating: int):
    db.execute("""
        INSERT INTO track_scores (track_title, track_artist, mood, thumbs_up, thumbs_down)
        VALUES (?, ?, ?, 0, 0)
        ON CONFLICT(track_title, track_artist, mood) DO NOTHING
    """, (title, artist, mood))
    if rating == 1:
        db.execute("""
            UPDATE track_scores SET thumbs_up = thumbs_up + 1
            WHERE track_title = ? AND track_artist = ? AND mood = ?
        """, (title, artist, mood))
    else:
        db.execute("""
            UPDATE track_scores SET thumbs_down = thumbs_down + 1
            WHERE track_title = ? AND track_artist = ? AND mood = ?
        """, (title, artist, mood))
    db.execute("""
        UPDATE track_scores
        SET score = CAST(thumbs_up - thumbs_down AS REAL) / (thumbs_up + thumbs_down)
        WHERE track_title = ? AND track_artist = ? AND mood = ?
    """, (title, artist, mood))
    db.commit()

def get_track_scores(mood_tags: list) -> dict:
    scores = {}
    for mood in mood_tags:
        rows = db.execute("""
            SELECT track_title, track_artist, score, thumbs_up, thumbs_down
            FROM track_scores
            WHERE mood = ? AND (thumbs_up + thumbs_down) >= 2
            ORDER BY score DESC
        """, (mood,)).fetchall()
        for row in rows:
            key = f"{row[0]}|||{row[1]}"
            scores[key] = {
                "score": row[2],
                "thumbs_up": row[3],
                "thumbs_down": row[4]
            }
    return scores



def rerank_tracks(tracks: list, mood_tags: list) -> list:
    scores = get_track_scores(mood_tags)
    if not scores:
        return tracks
    for track in tracks:
        key = f"{track['title']}|||{track['artist']}"
        if key in scores:
            track["community_score"] = scores[key]["score"]
            track["thumbs_up"] = scores[key]["thumbs_up"]
            track["thumbs_down"] = scores[key]["thumbs_down"]
        else:
            track["community_score"] = 0.0
    scored = [t for t in tracks if t.get("thumbs_up", 0) + t.get("thumbs_down", 0) >= 2]
    unscored = [t for t in tracks if t.get("thumbs_up", 0) + t.get("thumbs_down", 0) < 2]
    scored.sort(key=lambda x: x["community_score"], reverse=True)
    return scored + unscored

# ─── MODELS ───────────────────────────────────────────────────

class SceneInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    track_title: str
    track_artist: str
    mood_tags: list
    language: str
    rating: int

# ─── GROQ ─────────────────────────────────────────────────────

def call_groq_mood(text: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You extract music mood attributes from scene descriptions.
The input may be in Tamil, English, Hinglish, or any mix — understand it as-is.
Also detect the primary language of the input and include it.
Return ONLY valid JSON, no explanation, no markdown:
{
  "mood": ["string"],
  "energy": 0.0,
  "valence": 0.0,
  "era": ["string"],
  "genre": ["string"],
  "tempo_feel": "slow|mid|upbeat",
  "playlist_name": "string",
  "input_language": "Tamil|Hindi|English|Mixed"
}
energy and valence are 0.0 to 1.0. playlist_name should be poetic, lowercase."""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.7,
        max_tokens=300,
    )
    raw = response.choices[0].message.content
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    if not raw.endswith("}"):
        raw = raw + "}"
    return json.loads(raw)

def clean_transcript(raw_transcript: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a speech transcript cleaner.
Fix STT errors, remove filler words, keep original language and emotional meaning.
Return ONLY the cleaned transcript text, nothing else."""
            },
            {
                "role": "user",
                "content": raw_transcript
            }
        ],
        temperature=0.3,
        max_tokens=200,
    )
    cleaned = response.choices[0].message.content.strip()
    print(f"ORIGINAL: {raw_transcript}")
    print(f"CLEANED:  {cleaned}")
    return cleaned

def call_groq_song_recommendations(mood_data: dict, scene_text: str) -> list:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a music expert who recommends real, specific songs based on mood and context.

LANGUAGE RULES:
- If input is in Tamil → recommend 5 Tamil songs first, then 3 from other languages
- If input is in Hindi/Hinglish → recommend 5 Hindi songs first, then 3 others
- If input is in English → mix all languages equally (Tamil, Hindi, English)
- Always include regional Indian music when mood fits

ERA/AGE RULES:
- "old", "classic", "90s", "childhood", "paati" → Ilaiyaraaja, AR Rahman 90s, SPB, KJ Yesudas
- "new", "recent", "latest" → Anirudh, Sid Sriram, Sai Abhyankkar
- Default: 40% classics + 60% contemporary
- 80s/90s Tamil → Ilaiyaraaja
- 2000s Tamil → AR Rahman Tamil, Harris Jayaraj, Yuvan
- 2010s+ Tamil → Anirudh, Sid Sriram, D. Imman
- Hindi classics → Kishore Kumar, Lata, RD Burman
- Hindi modern → Arijit Singh, Pritam

MOOD MATCHING:
- Rainy/nostalgic → Ilaiyaraaja rainy classics, AR Rahman monsoon songs
- Happy/energetic → upbeat Anirudh, peppy Yuvan
- Sad/missing → Sid Sriram, Harris Jayaraj, Arijit Singh
- Late night → Sid Sriram, Sai Abhyankkar, AR Rahman ambient
- Romantic → Harris Jayaraj, AR Rahman, Pritam

Return ONLY a JSON array of 8 objects:
[
  {
    "title": "exact song title",
    "artist": "exact artist name",
    "language": "Tamil/Hindi/English/Telugu",
    "why": "one poetic line why this fits (max 8 words)"
  }
]"""
            },
            {
                "role": "user",
                "content": f"Scene: {scene_text}\nMood: {json.dumps(mood_data)}\nLanguage: {mood_data.get('input_language', 'English')}"
            }
        ],
        temperature=0.8,
        max_tokens=800,
    )
    raw = response.choices[0].message.content
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except:
        return []

# ─── ITUNES ───────────────────────────────────────────────────

def fetch_itunes_for_song(title: str, artist: str) -> dict:
    query = f"{title} {artist}"
    r = req_lib.get("https://itunes.apple.com/search", params={
        "term": query,
        "media": "music",
        "entity": "song",
        "limit": 3,
        "country": "IN"
    })
    results = r.json().get("results", [])
    if results:
        t = results[0]
        return {
            "title": t["trackName"],
            "artist": t["artistName"],
            "preview_url": t.get("previewUrl"),
            "spotify_url": t.get("trackViewUrl"),
            "album_art": t.get("artworkUrl100")
        }
    else:
        yt_query = f"{title} {artist}".replace(" ", "+")
        return {
            "title": title,
            "artist": artist,
            "preview_url": None,
            "spotify_url": f"https://www.youtube.com/results?search_query={yt_query}",
            "album_art": None
        }

# ─── ROUTES ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "vibe-playlist backend running"}

@app.get("/debug")
def debug():
    return {"groq_key_set": bool(os.environ.get("GROQ_API_KEY"))}

@app.post("/extract-mood")
async def extract_mood(scene: SceneInput):
    if not scene.text.strip():
        raise HTTPException(status_code=400, detail="Scene description cannot be empty")
    try:
        return call_groq_mood(scene.text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM returned invalid JSON, try again")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get-playlist")
async def get_playlist(scene: SceneInput):
    if not scene.text.strip():
        raise HTTPException(status_code=400, detail="Empty input")
    try:
        mood_data = call_groq_mood(scene.text)
        print("MOOD:", mood_data)
        recommended = call_groq_song_recommendations(mood_data, scene.text)
        print("RECOMMENDED:", recommended)
        tracks = []
        for song in recommended:
            track = fetch_itunes_for_song(song["title"], song["artist"])
            track["why"] = song.get("why", "")
            track["language"] = song.get("language", "")
            tracks.append(track)
        tracks = rerank_tracks(tracks, mood_data.get("mood", []))
        return {"mood_data": mood_data, "tracks": tracks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackInput):
    if feedback.rating not in [1, -1]:
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")
    try:
        for mood in feedback.mood_tags:
            db.execute("""
                INSERT INTO feedback (track_title, track_artist, mood, language, rating)
                VALUES (?, ?, ?, ?, ?)
            """, (feedback.track_title, feedback.track_artist, mood,
                  feedback.language, feedback.rating))
            update_track_score(
                feedback.track_title,
                feedback.track_artist,
                mood,
                feedback.rating
            )
        db.commit()
        return {"status": "ok", "message": "feedback saved"}
    except Exception as e:
        print("FEEDBACK ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    rows = db.execute("""
        SELECT track_title, track_artist, mood, thumbs_up, thumbs_down, score
        FROM track_scores
        WHERE (thumbs_up + thumbs_down) >= 1
        ORDER BY score DESC
        LIMIT 20
    """).fetchall()
    return {
        "top_tracks": [
            {
                "title": r[0], "artist": r[1], "mood": r[2],
                "thumbs_up": r[3], "thumbs_down": r[4],
                "score": round(r[5], 2)
            }
            for r in rows
        ]
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    try:
        temp_path = f"temp_audio_{file.filename}"
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        with open(temp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="text",
                **({"language": language} if language != "auto" else {})
            )
        os.remove(temp_path)
        cleaned = clean_transcript(transcription)
        return {"transcript": cleaned, "raw_transcript": transcription}
    except Exception as e:
        print("Transcription error:", e)
        raise HTTPException(status_code=500, detail=str(e))
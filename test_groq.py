import os, json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def extract_mood(scene_description: str):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You extract music mood attributes from scene descriptions.
The input may be in Tamil, English, Hinglish, or any mix — understand it as-is.
Return ONLY valid JSON, no explanation, no markdown:
{
  "mood": ["string"],
  "energy": 0.0,
  "valence": 0.0,
  "era": ["string"],
  "genre": ["string"],
  "tempo_feel": "slow|mid|upbeat",
  "playlist_name": "string"
}
energy and valence are 0.0 to 1.0. playlist_name should be poetic, lowercase."""
            },
            {
                "role": "user",
                "content": scene_description
            }
        ],
        temperature=0.7,
        max_tokens=300,
    )
    
    raw = response.choices[0].message.content
    return json.loads(raw)

# test it
result = extract_mood("rainy monday morning in a café, feeling nostalgic and a bit lonely")
print(json.dumps(result, indent=2))

# test Tamil/mixed
result2 = extract_mood("illa pa romba tired ah irukken, rain also coming outside, sad mood")
print(json.dumps(result2, indent=2))
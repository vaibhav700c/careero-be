from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import tempfile
from dotenv import load_dotenv
import google.generativeai as genai
import assemblyai as aai
from elevenlabs import ElevenLabs
import json
import traceback
import time
from datetime import datetime, timedelta

# ------------------ Load environment variables ------------------
ENV_PATH = ".env"   # .env file in the same directory
load_dotenv(dotenv_path=ENV_PATH)

ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

if not (ASSEMBLYAI_KEY and GEMINI_KEY and ELEVEN_KEY):
    raise RuntimeError(
        "Missing API keys in .env (ASSEMBLYAI_API_KEY, GEMINI_API_KEY, ELEVENLABS_API_KEY)"
    )

# ------------------ Configure APIs ------------------
aai.settings.api_key = ASSEMBLYAI_KEY
genai.configure(api_key=GEMINI_KEY)
eleven_client = ElevenLabs(api_key=ELEVEN_KEY)

# ------------------ Rate limiting and quota tracking ------------------
class TTSQuotaTracker:
    def __init__(self):
        self.request_times = []
        self.quota_exhausted = False
        self.quota_reset_time = None
        self.max_requests_per_minute = 20  # Conservative limit for free tier
        
    def can_make_request(self):
        """Check if we can make a request without hitting rate limits"""
        now = datetime.now()
        
        # If quota is exhausted and we haven't reached reset time, return False
        if self.quota_exhausted and self.quota_reset_time and now < self.quota_reset_time:
            return False
            
        # Clean old requests (older than 1 minute)
        self.request_times = [req_time for req_time in self.request_times 
                             if now - req_time < timedelta(minutes=1)]
        
        # Check if we're under the rate limit
        return len(self.request_times) < self.max_requests_per_minute
    
    def record_request(self):
        """Record a successful request"""
        self.request_times.append(datetime.now())
        
    def record_quota_exhausted(self):
        """Record that quota is exhausted, set reset time to 1 hour from now"""
        self.quota_exhausted = True
        self.quota_reset_time = datetime.now() + timedelta(hours=1)
        print(f"ElevenLabs quota exhausted, will retry after {self.quota_reset_time}")
        
    def reset_quota_status(self):
        """Reset quota status if enough time has passed"""
        if self.quota_exhausted and self.quota_reset_time and datetime.now() >= self.quota_reset_time:
            self.quota_exhausted = False
            self.quota_reset_time = None
            print("ElevenLabs quota status reset, attempting API calls again")

tts_quota_tracker = TTSQuotaTracker()

# ------------------ FastAPI app ------------------
app = FastAPI(title="AI Career Assessment API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://neuro-career-844x.vercel.app",  # Your actual Vercel domain
        "https://neuro-career-844x-git-main-vaibhav700cs-projects.vercel.app",  # Git branch preview URLs
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Pydantic models ------------------
class ChatRequest(BaseModel):
    message: str

class TTSRequest(BaseModel):
    message: str

class TTSStatusResponse(BaseModel):
    can_use_elevenlabs: bool
    requests_remaining: int
    quota_reset_time: str = None

# ------------------ State (use consistent slot keys) ------------------
state = {
    "Age": None,
    "School Class": None,
    "Location": None,
    "Interests": None,
    "Skills": None,
    "Constraints": None,
    "Values": None,
    "Prior Exploration": None,
}

# ------------------ AI Assistant ------------------
class AI_Assistant:
    def __init__(self):
        # model name kept as in your original code; change if needed
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.system_prompt = """
You are a prototype career counsellor chatbot named Lonita.
Your role is to help students explore career paths based on their basic background and preferences.

You MUST always return valid JSON with two keys:
- "slots": a dictionary where keys are the fields (Age, School Class, Location, Interests, Skills, Constraints, Values, Prior Exploration)
    that the student answered in this turn, and values are their responses. If none, return {}.
- "response": your reply to the student.

Never ask again about fields that are already filled in the state dictionary.

Rules:
1. Collect answers for the following fields (all questions at once, in natural conversation):
   - Age
   - School class
   - Location
   - Interests
   - Skills
   - Constraints
   - Values
   - Prior exploration

2. Never repeat questions about the same field once it is filled.
3. Use the provided context prompt to guide you.

When all fields are filled, return that the optimal career path for you is Software Engineering and direct them to go to Simulations page of the website to start their VR experience.

Return ONLY valid JSON, nothing else.
"""

    def process_new_answers(self, slots: dict):
        """
        Update state with the provided slots dictionary.
        Only fills a slot if the state slot is currently None or empty.
        """
        if not isinstance(slots, dict):
            return
        for slot_key, slot_value in slots.items():
            if not slot_key:
                continue
            # Accept either exact slot keys or case-insensitive match
            for canonical in list(state.keys()):
                if slot_key.strip().lower() == canonical.lower():
                    if not state[canonical]:
                        state[canonical] = slot_value
                    break

    def build_context_prompt(self):
        prompt = "Here is what we've learned so far:\n"
        for k, v in state.items():
            prompt += f"- {k}: {v if v else '<missing>'}\n"
        prompt += (
            "\nAsk only about missing fields. Ask about only one or a small set of missing fields "
            "at a time, and do NOT repeat questions for fields that are already filled."
        )
        return prompt

    def get_response(self, user_input: str) -> dict:
        """
        Returns a dict with:
        {
            "slots": { "Age": "17", "Skills": "coding, python" },
            "response": "Acknowledging message and next question"
        }
        This tries to parse model output as JSON. If model returns non-JSON, we fallback safely.
        """
        try:
            context_prompt = self.build_context_prompt()
            full_prompt = f"{self.system_prompt}\n\n{context_prompt}\nStudent says: \"{user_input}\"\n\nJSON:"

            # call model
            response = self.model.generate_content(full_prompt)

            # If response object has text attr, try to parse it
            text_out = None
            if response is None:
                text_out = None
            elif hasattr(response, "text") and response.text:
                text_out = response.text
            else:
                # Some SDK variants put content in candidates/parts; attempt safe extraction
                try:
                    # defensive access
                    if hasattr(response, "candidates") and response.candidates:
                        # join candidate texts if present
                        parts = []
                        for cand in response.candidates:
                            if hasattr(cand, "content"):
                                # content might be nested
                                content = getattr(cand, "content")
                                if isinstance(content, list):
                                    for p in content:
                                        if hasattr(p, "text"):
                                            parts.append(p.text)
                                elif hasattr(content, "text"):
                                    parts.append(content.text)
                        if parts:
                            text_out = "\n".join(parts)
                except Exception:
                    text_out = None

            if not text_out:
                # safe default: no slots, simple question back
                return {
                    "slots": {},
                    "response": "I'm here to help with your career exploration. Could you tell me a bit about yourself (age, class, interests, skills, constraints, values, prior exploration)?"
                }

            # Try to parse JSON from text_out
            try:
                cleaned = text_out.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")
                    # Remove optional "json" after ```
                    if cleaned.lower().startswith("json"):
                        cleaned = cleaned[4:].strip()
                
                parsed = json.loads(cleaned)
                # Ensure dict has required keys
                if not isinstance(parsed, dict):
                    raise ValueError("Parsed not a dict")
                # ensure both keys exist (provide defaults)
                slots = parsed.get("slots", {})
                resp = parsed.get("response", "")
                # normalize slots to dict if None
                if slots is None:
                    slots = {}
                
                return {"slots": slots, "response": resp}
            except Exception:
                # If model returned non-JSON text, safely fall back to returning that text as response and no slots
                return {"slots": {}, "response": text_out.strip()}

        except Exception as e:
            traceback.print_exc()
            return {"slots": {}, "response": "I apologize, but I'm having trouble responding right now. Could you please try again?"}
        



# Initialize AI Assistant
assistant = AI_Assistant()

# ------------------ Endpoints ------------------
@app.get("/")
async def root():
    return {"message": "AI Career Assessment API is running!"}


@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe uploaded audio file using AssemblyAI.
       Note: AssemblyAI SDKs vary in interface. The following uses Transcriber().transcribe(file_path)
       which works for many versions — if your version requires upload + polling, replace accordingly.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        audio_data = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        try:
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(temp_file_path)

            # Several SDKs return transcript.text or transcript.content
            transcript_text = ""
            try:
                transcript_text = getattr(transcript, "text", None) or getattr(transcript, "content", None) or ""
            except Exception:
                transcript_text = str(transcript)

            if not transcript_text:
                transcript_text = "No speech detected in the audio."

            return {"transcription": transcript_text}

        finally:
            # Clean up
            if os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Hybrid chat endpoint:
    - First, ask Gemini for structured JSON: {"slots": {...}, "response": "..."}
    - If Gemini returns no slots, use keyword fallback detection (if/elif chain) to attempt to extract obvious fields
    - Update the global state with any newly-detected slots (only fills empty slots)
    - Return the assistant response plus updated state
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        user_msg = request.message.strip()
        ai_reply = assistant.get_response(user_msg)

        # Expect ai_reply to be dict with 'slots' and 'response'
        slots = ai_reply.get("slots", {}) if isinstance(ai_reply, dict) else {}
        response_text = ai_reply.get("response", "") if isinstance(ai_reply, dict) else str(ai_reply)

        # --- Fallback keyword detection (only if model returned no slots) ---
        # We'll attempt to find multiple slots in a single user message (not just one)
        if not slots:
            low = user_msg.lower()
            candidate_slots = {}

            # Age
            # look for patterns like "i am 17", "i'm 17", "17 years", or "17-year-old"
            import re
            age_match = re.search(r"\b(?:i am|i'm|i’m|age is|age)\s+(\d{1,2})\b", low)
            if not age_match:
                # also numbers followed by 'years' or 'years old'
                age_match = re.search(r"\b(\d{1,2})\s+(?:years|yrs|years old|yrs old)\b", low)
            if age_match:
                candidate_slots["Age"] = age_match.group(1)

            # School Class / grade
            if any(word in low for word in ["12th", "11th", "10th", "grade", "class"]):
                # naive extraction: take the whole message as class if class/grade mentioned
                # you could refine this with regex
                if "12th" in low:
                    candidate_slots["School Class"] = "12th"
                elif "11th" in low:
                    candidate_slots["School Class"] = "11th"
                elif "10th" in low:
                    candidate_slots["School Class"] = "10th"
                else:
                    # fallback: the exact phrase containing 'class' or 'grade'
                    candidate_slots["School Class"] = user_msg

            # Location
            if any(word in low for word in ["city", "town", "live in", "i live in", "from"]):
                # simple heuristic: assume last word(s)
                candidate_slots["Location"] = user_msg

            # Interests
            if any(word in low for word in ["interest", "interested", "i like", "i love", "i'm interested in", "interested in"]):
                candidate_slots["Interests"] = user_msg

            # Skills
            if any(word in low for word in ["skill", "good at", "i can", "i'm good at", "i am good at", "coding", "python", "java", "programming"]):
                candidate_slots["Skills"] = user_msg

            # Constraints
            if any(word in low for word in ["constraint", "constraints", "parents", "can't", "cannot", "need to stay", "stay in"]):
                candidate_slots["Constraints"] = user_msg

            # Values
            if any(word in low for word in ["value", "values", "work-life", "work life", "helping others", "money", "balance"]):
                candidate_slots["Values"] = user_msg

            # Prior Exploration
            if any(word in low for word in ["hackathon", "intern", "internship", "project", "tried", "explored", "experience"]):
                candidate_slots["Prior Exploration"] = user_msg

            # Use candidate_slots if any found
            if candidate_slots:
                slots = candidate_slots

        # Update global state with whatever slots we detected
        if isinstance(slots, dict) and slots:
            assistant.process_new_answers(slots)

        return {
            "response": ai_reply.get("response", ""),
            "state": state
        }


    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/api/text-to-speech")
async def text_to_speech(request: TTSRequest):
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Extract text to convert from the message
        text_to_convert = request.message.strip()
        
        # Try to parse as JSON first (in case it's a structured response)
        try:
            parsed_json = json.loads(text_to_convert)
            if isinstance(parsed_json, dict) and "response" in parsed_json:
                text_to_convert = parsed_json["response"]
        except (json.JSONDecodeError, TypeError):
            # If it's not JSON, use the original text as is
            pass
        
        # Ensure we have text to convert
        if not text_to_convert or not text_to_convert.strip():
            raise HTTPException(status_code=400, detail="No text found to convert to speech")

        print(f"Converting to speech: {text_to_convert[:100]}...")  # Log for debugging

        # Check quota status first
        tts_quota_tracker.reset_quota_status()
        
        # If quota is exhausted or rate limit exceeded, return 429 immediately
        if not tts_quota_tracker.can_make_request():
            print("Rate limit or quota exceeded, returning 429 immediately")
            raise HTTPException(
                status_code=429, 
                detail="ElevenLabs API quota exceeded or rate limit reached. Please use browser speech synthesis fallback."
            )

        try:
            # Get audio stream from ElevenLabs
            audio_stream = eleven_client.text_to_speech.convert(
                voice_id="pNInz6obpgDQGcFmaJgB",
                text=text_to_convert,
                model_id="eleven_turbo_v2"
            )

            # Collect chunks into final bytes
            audio_bytes = b"".join(chunk for chunk in audio_stream if isinstance(chunk, (bytes, bytearray)))

            # Record successful request
            tts_quota_tracker.record_request()

            return Response(
                content=audio_bytes,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=speech.mp3",
                    "X-TTS-Source": "elevenlabs"  # Header to indicate source
                },
            )
        
        except Exception as elevenlabs_error:
            # Check if it's an API quota/auth error
            error_str = str(elevenlabs_error).lower()
            
            # More comprehensive error detection including unusual activity
            quota_keywords = ['401', '429', 'quota', 'unusual activity', 'unusual_activity', 
                            'detected_unusual_activity', 'free tier', 'rate limit', 
                            'too many requests', 'usage limit', 'exceeded', 'abuse', 'disabled']
            
            if any(keyword in error_str for keyword in quota_keywords):
                print(f"ElevenLabs API blocked (quota/auth/abuse issue): {elevenlabs_error}")
                
                # Record quota exhaustion for longer period if it's an abuse detection
                if 'unusual' in error_str or 'abuse' in error_str or 'disabled' in error_str:
                    print("Detected ElevenLabs abuse/unusual activity - disabling for extended period")
                    tts_quota_tracker.quota_exhausted = True
                    tts_quota_tracker.quota_reset_time = datetime.now() + timedelta(hours=24)  # 24 hour cooldown
                else:
                    tts_quota_tracker.record_quota_exhausted()
                
                # Return a 429 status to trigger frontend fallback to browser speech synthesis
                raise HTTPException(
                    status_code=429, 
                    detail="ElevenLabs API quota exceeded or unusual activity detected. Using browser speech synthesis fallback."
                )
            else:
                # Re-raise for other types of errors
                print(f"ElevenLabs API error (non-quota): {elevenlabs_error}")
                raise elevenlabs_error

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")


@app.get("/api/tts-status")
async def get_tts_status():
    """Get current TTS quota status to help frontend decide whether to use ElevenLabs or browser speech"""
    try:
        tts_quota_tracker.reset_quota_status()
        can_use = tts_quota_tracker.can_make_request()
        
        remaining_requests = max(0, tts_quota_tracker.max_requests_per_minute - len(tts_quota_tracker.request_times))
        
        reset_time = None
        if tts_quota_tracker.quota_reset_time:
            reset_time = tts_quota_tracker.quota_reset_time.isoformat()
            
        return TTSStatusResponse(
            can_use_elevenlabs=can_use and not tts_quota_tracker.quota_exhausted,
            requests_remaining=remaining_requests,
            quota_reset_time=reset_time
        )
    except Exception as e:
        # If there's an error checking status, assume we can't use ElevenLabs
        return TTSStatusResponse(
            can_use_elevenlabs=False,
            requests_remaining=0,
            quota_reset_time=None
        )


# ------------------ Run server ------------------
if __name__ == "__main__":
    import os
    
    print("Starting AI Career Assessment FastAPI server...")
    print("Available endpoints:")
    print("  GET  / - Health check")
    print("  POST /api/transcribe - Audio transcription")
    print("  POST /api/chat - AI chat responses")
    print("  POST /api/text-to-speech - Text-to-speech conversion")
    print("  GET /api/tts-status - TTS quota and availability status")
    
    # Get port from environment (for Railway, Heroku, etc.) or default to 8000
    port = int(os.getenv("PORT", 8000))
    print(f"\nServer running at: http://0.0.0.0:{port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

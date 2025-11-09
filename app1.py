import os
import queue
import threading
import tempfile
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

import assemblyai as aai
import google.generativeai as genai
from elevenlabs import ElevenLabs

# ---------------- CONFIG ----------------
ENV_PATH = r"C:\full_prototype\Neuro-Career\neuro-career-be\.env"   # change if needed
SAMPLE_RATE = 16000
BLOCKSIZE = 1024                        # input block size for sounddevice callback
SILENCE_THRESHOLD = 0.01                # RMS threshold to consider "speech"
SILENCE_DURATION = 0.7                  # seconds of silence to mark end of utterance
MIN_UTTERANCE_DURATION = 0.15           # ignore very short noises (< seconds)
ELEVEN_VOICE_ID = "pNInz6obpgDQGcFmaJgB"
ELEVEN_TTS_MODEL = "eleven_turbo_v2"
CUSTOM_PROMPT = """
You are a prototype career counsellor chatbot named Lonita. 
Your role is to help students explore career paths based on their basic background and preferences. 

ASk the following questions all at once: 
   - Age  
   - School class  
   - Location  
   - Interests  
   - Skills  
   - Constraints (e.g., parental preferences, time constraints)  
   - Values (e.g., helping others, creativity, money, work-life balance)  
   - Prior exploration (activity, description, duration, feedback)  
Once the user has answered all questions, reply with Software engineer as the optimal career path
7. Remember: you are a prototype, so the final career path is always Software Engineer, regardless of inputs
"""

prompt = "Here is what we've learned so far:\n"

state = {
    "age": None,
    "Class": None,
    "Location": None,
    "Interests": None,
    "SkillsConstraints": None,
    "Values": None,
    "PriorExploration": None
}

# ---------------- SETUP KEYS ----------------
load_dotenv(dotenv_path=ENV_PATH)
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")

if not (ASSEMBLYAI_KEY and GEMINI_KEY and ELEVEN_KEY):
    raise RuntimeError("Missing API keys in .env (ASSEMBLYAI_API_KEY, GEMINI_API_KEY, ELEVENLABS_API_KEY)")

aai.settings.api_key = ASSEMBLYAI_KEY
genai.configure(api_key=GEMINI_KEY)
eleven_client = ElevenLabs(api_key=ELEVEN_KEY)

# ---------------- ASSISTANT ----------------
class AI_Assistant:
    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        silence_threshold=SILENCE_THRESHOLD,
        silence_duration=SILENCE_DURATION
    ):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

        self.audio_queue = queue.Queue()
        self.recording = False
        self.speaking = False
        self.stream = None
        self.process_thread = None

        # For generating responses
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            print("InputStream status:", status)
        # Drop incoming audio while assistant is playing to avoid echo-back
        if self.speaking:
            return
        # copy to keep the numpy buffer valid
        self.audio_queue.put(indata.copy())

    def start_listening(self):
        if self.recording:
            print("Already listening.")
            return
        self.recording = True
        # Start input stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._mic_callback,
            blocksize=self.blocksize,
            dtype="float32"
        )
        self.stream.start()
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        print("Started listening. (Speak; assistant will auto-detect utterances.)")

    def stop_listening(self):
        self.recording = False
        # stop stream
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception:
            pass
        # wait for processing thread
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
        print("Stopped listening.")

    def _process_loop(self):
        """
        Read audio blocks from queue, use RMS to detect speech start/end.
        When speech segment detected, hand it off to handler thread.
        """
        buffer_blocks = []
        speech_active = False
        silence_time = 0.0
        last_time = time.time()

        while self.recording:
            try:
                block = self.audio_queue.get(timeout=0.3)  # block: numpy array shape (n,1)
            except queue.Empty:
                # if waiting and previously had some short speech, check silence timer
                if speech_active:
                    now = time.time()
                    silence_time += now - last_time
                    last_time = now
                    if silence_time >= self.silence_duration:
                        # finalize utterance
                        audio_np = np.concatenate(buffer_blocks, axis=0)
                        total_sec = audio_np.shape[0] / self.sample_rate
                        buffer_blocks = []
                        speech_active = False
                        silence_time = 0.0
                        if total_sec >= MIN_UTTERANCE_DURATION:
                            threading.Thread(target=self._handle_utterance, args=(audio_np,), daemon=True).start()
                        else:
                            # ignore too-short capture
                            pass
                continue

            # got a block
            now = time.time()
            last_time = now
            # compute RMS energy
            rms = np.sqrt(np.mean(np.square(block.astype("float32"))))
            if rms >= self.silence_threshold:
                # speech detected
                buffer_blocks.append(block)
                speech_active = True
                silence_time = 0.0
            else:
                # below threshold
                if speech_active:
                    buffer_blocks.append(block)
                    # accumulate silence
                    silence_time += block.shape[0] / self.sample_rate
                    if silence_time >= self.silence_duration:
                        # finalize
                        audio_np = np.concatenate(buffer_blocks, axis=0)
                        total_sec = audio_np.shape[0] / self.sample_rate
                        buffer_blocks = []
                        speech_active = False
                        silence_time = 0.0
                        if total_sec >= MIN_UTTERANCE_DURATION:
                            threading.Thread(target=self._handle_utterance, args=(audio_np,), daemon=True).start()
                        else:
                            # noise only
                            pass
                else:
                    # not in speech, drop silent block (keeps memory small)
                    pass

        # on exit, flush any buffered speech
        if buffer_blocks:
            audio_np = np.concatenate(buffer_blocks, axis=0)
            if audio_np.shape[0] / self.sample_rate >= MIN_UTTERANCE_DURATION:
                threading.Thread(target=self._handle_utterance, args=(audio_np,), daemon=True).start()


    def process_new_answer(slot, value):
        state[slot] = value
        
        for k, v in state.items():
            if v:
                prompt += f"- {k.capitalize()}: {v}\n"
        prompt += "Which of the required pieces of information is missing? Ask only one at a time, and do NOT repeat questions for fields already filled."
        return prompt

    def _handle_utterance(self, audio_np):
        """
        Save audio to temp wav, call AssemblyAI for transcription,
        call Gemini for reply, then TTS+playback.
        """
        # Save wav
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                wav_path = tf.name
            sf.write(wav_path, audio_np, self.sample_rate, subtype="PCM_16")
        except Exception as e:
            print("Error saving WAV:", e)
            return

        # Transcribe (AssemblyAI - synchronous batch)
        transcript = None
        try:
            transcriber = aai.Transcriber()
            result = transcriber.transcribe(wav_path)
            transcript = result.text if getattr(result, "text", None) else None
        except Exception as e:
            print("AssemblyAI error:", e)

        # cleanup input wav
        try:
            os.remove(wav_path)
        except Exception:
            pass

        if not transcript:
            print("[No speech recognized / transcription empty]")
            return

        print("\nUser:", transcript)

        # Generate AI reply (Gemini)
        try:
            full_prompt = CUSTOM_PROMPT.format(user_input=transcript)
            resp = self.model.generate_content(full_prompt)
            ai_reply = resp.text.strip() if getattr(resp, "text", None) else None
        except Exception as e:
            print("Gemini error:", e)
            ai_reply = "Sorry, I couldn't produce an answer right now."

        if not ai_reply:
            print("[No AI reply]")
            return

        print("Assistant:", ai_reply)

        # Generate TTS and play (pause listening during playback)
        try:
            self.speaking = True
            audio_iter = eleven_client.text_to_speech.convert(
                voice_id=ELEVEN_VOICE_ID,
                model_id=ELEVEN_TTS_MODEL,
                text=ai_reply
            )

            # write to temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf_out:
                out_path = tf_out.name
                for chunk in audio_iter:
                    if chunk:
                        tf_out.write(chunk)
            # attempt to read & play via sounddevice
            try:
                data, sr = sf.read(out_path, dtype="float32")
                sd.play(data, sr)
                sd.wait()
            except Exception as e:
                # fallback: on Windows try os.startfile (non-blocking)
                print("Direct playback failed, falling back to OS player:", e)
                try:
                    if os.name == "nt":
                        os.startfile(out_path)
                        # wait heuristically until file finishes: play length from soundfile if available
                        try:
                            info = sf.info(out_path)
                            wait_secs = info.duration if info.duration else 0
                            time.sleep(wait_secs + 0.3)
                        except Exception:
                            time.sleep(1.0)
                    else:
                        print("Please play the file manually:", out_path)
                        time.sleep(1.0)
                except Exception as e2:
                    print("Fallback playback also failed:", e2)
            # cleanup tts file
            try:
                os.remove(out_path)
            except Exception:
                pass
        finally:
            self.speaking = False

# ---------------- MAIN ----------------
if __name__ == "__main__":
    assistant = AI_Assistant()
    try:
        greeting = (
            "Hey there! Wonderful to have another enthusiast ready to explore the VR world of careers. "
            "My name is Lonita and I help analyze aptitudes to suggest career paths. Say 'yes' when you are ready."
        )
        # speak greeting synchronously
        # reuse same handler for TTS playback
        assistant._handle_utterance(np.zeros((int(SAMPLE_RATE*0.2),1), dtype="float32"))  # quick trick to create temp file path (not needed)
        # Instead of artificial call, directly generate greeting TTS:
        assistant.speaking = True
        audio_iter = eleven_client.text_to_speech.convert(
            voice_id=ELEVEN_VOICE_ID, model_id=ELEVEN_TTS_MODEL, text=greeting
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out_path = tf.name
            for chunk in audio_iter:
                if chunk:
                    tf.write(chunk)
        try:
            data, sr = sf.read(out_path, dtype="float32")
            sd.play(data, sr)
            sd.wait()
        except Exception:
            try:
                if os.name == "nt":
                    os.startfile(out_path)
                    # wait short time
                    time.sleep(1.0)
            except Exception:
                pass
        try:
            os.remove(out_path)
        except Exception:
            pass
        assistant.speaking = False

        # start continuous listening
        assistant.start_listening()
        print("Press Ctrl+C to quit.")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nReceived exit, shutting down...")
    finally:
        assistant.stop_listening()
        print("Goodbye.")

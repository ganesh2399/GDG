import os
import sys
import re
import random
from dotenv import load_dotenv

# Modern Gemini SDK imports
from google import genai
from google.genai import types

from flask import Flask, request, jsonify

# Speech recognition and TTS libraries
try:
    import speech_recognition as sr
except ImportError:
    raise ImportError("Please install: pip install SpeechRecognition pyaudio")

try:
    import pyttsx3
except ImportError:
    raise ImportError("Please install: pip install pyttsx3")

app = Flask(__name__)

# ------------------------------------------------------------
# Initialization: Gemini Client
# ------------------------------------------------------------
def init_gemini_client() -> genai.Client:
    """Initialize and return the Gemini Client."""
    load_dotenv()  # Load .env if present
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n❌ ERROR: GEMINI_API_KEY not set. Create a .env file with GEMINI_API_KEY=your_key")
        sys.exit(1)
        
    try:
        # The SDK automatically picks up GEMINI_API_KEY from the environment
        return genai.Client()
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialise Gemini client: {e}")
        sys.exit(1)

client = init_gemini_client()
MODEL_ID = 'gemini-2.5-flash'

# ------------------------------------------------------------
# Helper: Text-to-Speech Engine
# ------------------------------------------------------------
def speak(text: str):
    """Speak the given text aloud using a local TTS engine."""
    engine = pyttsx3.init()
    # Optional: Adjust voice properties for a friendlier CX tone
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id) # Usually a female voice, often preferred for CX
        
    engine.setProperty('rate', 160)  # Slightly slower, clear pace
    engine.say(text)
    engine.runAndWait()

# ------------------------------------------------------------
# CX Feature: Mock Order Status DB
# ------------------------------------------------------------
def check_order_status(order_id: str) -> str:
    """Return a mock status for a customer's order."""
    statuses = [
        "is currently out for delivery and should arrive by 8 PM today.",
        "has been shipped and is in transit.",
        "is currently being processed at our warehouse.",
        "was delivered successfully yesterday. If you haven't received it, please let me know."
    ]
    status = random.choice(statuses)
    return f"I checked our system for order {order_id}. Your package {status}"

# ------------------------------------------------------------
# Core Processing Logic (Used by both Flask and Voice loop)
# ------------------------------------------------------------
def process_customer_query(user_text: str) -> str:
    """Processes the text, checks for specific CX intents, or falls back to Gemini."""
    
    # 1. Intent Detection: Check for order tracking
    order_match = re.search(r"(order|tracking) (number )?(\d+)", user_text, re.I)
    if order_match:
        order_id = order_match.group(3)
        return check_order_status(order_id)

    # 2. Fallback to Gemini AI for general inquiries
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_text,
            config=types.GenerateContentConfig(
                temperature=0.4, # Lower temperature for more factual/consistent support
                system_instruction=(
                    "You are 'Aura', a helpful, polite, and empathetic customer service voice assistant. "
                    "Keep your answers concise, clear, and easy to understand when spoken out loud. "
                    "Do not use markdown formatting like asterisks or bullet points, as this text will be read by a text-to-speech engine."
                )
            )
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "I'm sorry, our system is experiencing a temporary glitch. Please try asking again."

# ------------------------------------------------------------
# Flask Endpoints (For Web/App Integration)
# ------------------------------------------------------------
@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_text = data.get("text", "").strip()
    
    if not user_text:
        return jsonify({"error": "Missing 'text' field"}), 400

    reply = process_customer_query(user_text)
    return jsonify({"reply": reply})

# ------------------------------------------------------------
# Local Voice-First Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    # If you want to run the Flask server instead, comment out this block
    # and use: app.run(debug=True, port=5000)
    
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    welcome_msg = "Welcome to Customer Support. I am Aura. How can I help you today? Say 'exit' to quit."
    print(f"🗣️ {welcome_msg}")
    speak(welcome_msg)
    
    while True:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("\nListening…", end=" ", flush=True)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print("Listening timed out. Waiting...")
                continue
                
        try:
            # Recognize speech
            user_input = recognizer.recognize_google(audio)
            print(f"\nYou said: {user_input}")
            
            if user_input.lower() in {"exit", "quit", "stop", "goodbye"}:
                goodbye_msg = "Thank you for contacting support. Goodbye!"
                print(f"Bot: {goodbye_msg}")
                speak(goodbye_msg)
                break
                
            # Process query and speak response
            reply = process_customer_query(user_input)
            print(f"Bot: {reply}")
            speak(reply)
            
        except sr.UnknownValueError:
            error_msg = "I didn't quite catch that. Could you please repeat?"
            print(f"\n⚠️ {error_msg}")
            speak(error_msg)
        except sr.RequestError as e:
            print(f"\n⚠️ Speech-Recognition service error: {e}")
            speak("I am having trouble connecting to my audio sensors.")

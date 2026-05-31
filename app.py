import os
import re
import random
from dotenv import load_dotenv

# Modern Gemini SDK import
from google import genai
from google.genai import types
from flask import Flask, request, jsonify

app = Flask(__name__)

# ------------------------------------------------------------
# Initialization: Gemini Client (Modern SDK)
# ------------------------------------------------------------
load_dotenv()  # Load .env locally if present
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("⚠️ WARNING: GEMINI_API_KEY environment variable is not set.")
    client = None
else:
    # New syntax for initializing the client
    client = genai.Client(api_key=api_key)

MODEL_ID = 'gemini-2.5-flash'

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
# Core Processing Logic
# ------------------------------------------------------------
def process_customer_query(user_text: str) -> str:
    """Processes the text, checks for specific CX intents, or falls back to Gemini."""
    
    # 1. Intent Detection: Check for order tracking
    order_match = re.search(r"(order|tracking) (number )?(\d+)", user_text, re.I)
    if order_match:
        order_id = order_match.group(3)
        return check_order_status(order_id)

    # Check if Gemini model successfully initialized
    if not client:
        return "I'm sorry, the AI service is currently unconfigured. Please check back later."

    # 2. Fallback to Gemini AI for general inquiries
    try:
        # New syntax for generating content and passing instructions
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=user_text,
            config=types.GenerateContentConfig(
                temperature=0.4,
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
@app.route("/")
def home():
    return jsonify({"status": "online", "message": "Aura Customer Support API is running."})

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_text = data.get("text", "").strip()
    
    if not user_text:
        return jsonify({"error": "Missing 'text' field"}), 400

    reply = process_customer_query(user_text)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    # Cloud environments inject the PORT dynamic variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

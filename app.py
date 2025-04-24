import streamlit as st
import requests
import torch
from PIL import Image
import io

from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai

# === API KEYS ===
GENAI_API_KEY = "AIzaSyA-m2pt91XgayRfAYaV3KFBZ-g6pTdb_BI"
GOOGLE_KG_API_KEY = "AIzaSyAsJyPU-W-IiNm525tyzdakLkFi0uXAdIY"
OCR_API_KEY = "K89917156688957"
PREDICTION_API = "https://fakenewsfilter.onrender.com/predict"

# === CONFIGURE APIs ===
genai.configure(api_key=GENAI_API_KEY)
service = build('kgsearch', 'v1', developerKey=GOOGLE_KG_API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# === Gemini AI Fact-Checker ===
def get_fact_check_verification(user_statement):
    prompt = f"""
    You are an AI fact-checking assistant. Categorize the given statement into one of the following categories:
    - ✅ True: If the statement is entirely correct.
    - ❌ False: If the statement is incorrect or contradicts known facts.
    - 🤔 Likely True: If the statement is mostly correct but lacks some details.
    - ⚠️ Likely False: If the statement is misleading or lacks proper context.

    Statement: "{user_statement}"
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# === Image to Text using OCR ===
def image_to_text(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    response = requests.post(
        "https://api.ocr.space/parse/image",
        files={"file": ("image.png", img_bytes.getvalue())},
        data={"apikey": OCR_API_KEY, "language": "eng"},
    )
    result = response.json()
    if result["OCRExitCode"] == 1:
        return result["ParsedResults"][0]["ParsedText"].strip()
    return "Error: Unable to extract text."

# === Fake News Prediction ===
def predict_misinformation(text):
    payload = {"input": text}
    response = requests.post(PREDICTION_API, json=payload)
    if response.status_code == 200:
        return response.json()["prediction"]
    return f"Error: {response.status_code}, {response.text}"

# === Knowledge Graph Entity Search ===
def get_google_kg_entity(query):
    response = service.entities().search(query=query, limit=1).execute()
    if 'itemListElement' in response:
        entity = response['itemListElement'][0]
        result = entity['result']
        return result.get('name'), result.get('description', "No description"), result.get('url', None)
    return None, None, None

# === Similarity Check ===
def compute_similarity(input_text, page_content):
    input_embedding = embedder.encode([input_text])
    page_embedding = embedder.encode([page_content])
    return cosine_similarity(input_embedding, page_embedding)[0][0]

# === Knowledge Graph Based Fact Check ===
def user_friendly_fact_check(input_text, threshold=0.75):
    try:
        key_terms = input_text.split()[:3]
        best_match_score = 0
        best_match_details = None

        for term in key_terms:
            try:
                name, description, url = get_google_kg_entity(term)
                if name:
                    similarity_score = compute_similarity(input_text, description)
                    if similarity_score > best_match_score:
                        best_match_score = similarity_score
                        best_match_details = {"name": name, "description": description, "url": url}
            except Exception:
                continue

        if best_match_score > threshold:
            result = "✅ Fact Check Passed"
            confidence = "High Confidence"
            recommendation = "No further verification needed."
        elif best_match_score > 2 * threshold / 3:
            result = "⚠️ Likely True"
            confidence = "Moderate Confidence"
            recommendation = "Verify further using reliable sources."
        else:
            result = "❌ Likely False"
            confidence = "Low Confidence"
            recommendation = "Check with multiple sources for accuracy."

        return {
            "Fact-Check Result": result,
            "Entity Name": best_match_details['name'] if best_match_details else "No match found",
            "Description": best_match_details['description'][:300] + "..." if best_match_details else "No description available",
            "Confidence Level": confidence,
            "Similarity Score": round(best_match_score, 2),
            "Next Step": recommendation,
            "Entity URL": best_match_details['url'] if best_match_details else "No URL available"
        }

    except Exception as e:
        return {
            "Fact-Check Result": "⚠️ Error",
            "Message": "An error occurred during fact-checking.",
            "Error Details": str(e)
        }

# === Streamlit UI ===
st.set_page_config(page_title="🔍 Misinformation Detection", layout="wide")
st.markdown("# 🕵️ Misinformation Detection and Fact-Checking")
st.sidebar.markdown("### Advanced Options")
threshold = st.sidebar.slider("Set Similarity Threshold", 0.0, 1.0, 0.75)

uploaded_file = st.file_uploader("Upload an image for text extraction", type=["png", "jpg", "jpeg"])
user_input = ""
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    extracted_text = image_to_text(image)
    st.subheader("Extracted Text")
    st.write(extracted_text)
    user_input = extracted_text
else:
    user_input = st.text_area("Enter text to check", placeholder="Type or paste text to analyze...", height=200)

if st.button("Check News"):
    if user_input:
        with st.spinner("Analyzing news..."):
            prediction = predict_misinformation(user_input)
            if prediction == 1:
                st.success("✅ This news is real.")
            else:
                st.error("❌ This news is fake.")
    else:
        st.warning("Please enter some text or upload an image.")

if st.button("Check Facts"):
    if user_input:
        with st.spinner("Fact-checking..."):
            # --- Knowledge Graph ---
            result = user_friendly_fact_check(user_input, threshold)
            st.markdown("### 🧠 Knowledge Graph Based Check")
            st.markdown(f"**Fact-Check Result:** {result['Fact-Check Result']}")
            st.markdown(f"**Confidence Level:** {result['Confidence Level']}")
            st.markdown(f"**Similarity Score:** {result['Similarity Score']}")
            if result['Entity URL'] != "No URL available":
                st.markdown(f"[More Info]({result['Entity URL']})")
            st.markdown("---")

            # --- Gemini AI ---
            st.markdown("### 🤖 Gemini AI Based Check")
            gemini_result = get_fact_check_verification(user_input)
            st.markdown(f"**Gemini Verdict:** {gemini_result}")
    else:
        st.warning("Please enter some text or upload an image.")

with st.expander("ℹ️ How to Use"):
    st.write("""
        - Enter the text or upload an image to verify.
        - Click "Check News" for fake news detection.
        - Click "Check Facts" to fact-check using AI and the Knowledge Graph.
        - Use the threshold slider in the sidebar to control sensitivity of fact-checking.
    """)

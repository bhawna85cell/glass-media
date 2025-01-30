import streamlit as st
import requests
# from transformers import BertTokenizer, BertForSequenceClassification
import torch
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io

API_URL = "https://fakenewsfilter.onrender.com/predict"

embedder = SentenceTransformer('all-MiniLM-L6-v2')
API_KEY = 'AIzaSyC8SxCy92vwF0gpjZ2RF7uyolVcKaUJjMc'
service = build('kgsearch', 'v1', developerKey=API_KEY)
OCR_API_KEY = "K89917156688957"  # Replace with your OCRSpace API Key

def predict_misinformation(text):
    payload = {"input": text}
    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        return f"Error: {response.status_code}, {response.text}"
# Define OCR function using OCRSpace API
def image_to_text(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    response = requests.post(
        "https://api.ocr.space/parse/image",
        files={"file": ("image.png", img_bytes)},
        data={"apikey": OCR_API_KEY, "language": "eng"},
    )
    result = response.json()
    if result["OCRExitCode"] == 1:
        return result["ParsedResults"][0]["ParsedText"].strip()
    return "Error: Unable to extract text."

# Misinformation detection function


# Google Knowledge Graph Fact-Checking
def get_google_kg_entity(query):
    response = service.entities().search(query=query, limit=1).execute()
    if 'itemListElement' in response:
        entity = response['itemListElement'][0]
        entity_name = entity['result']['name']
        entity_description = entity['result']['description'] if 'description' in entity['result'] else "No description available"
        entity_url = entity['result']['url'] if 'url' in entity['result'] else None
        return entity_name, entity_description, entity_url
    return None, None, None

def compute_similarity(input_text, page_content):
    input_embedding = embedder.encode([input_text])
    page_embedding = embedder.encode([page_content])
    return cosine_similarity(input_embedding, page_embedding)[0][0]

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
            except Exception as e:
                print(f"Error fetching data for term '{term}': {e}")
                continue  # Skip the term and move to the next

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
            "Message": "An error occurred while processing the request. Please try again later.",
            "Error Details": str(e)  # Optional: You can remove this in production for security
        }


# Streamlit App UI
st.set_page_config(page_title="🔍Misinformation Detection", layout="wide")
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
                st.success("This news is real.")
            else:
                st.error("This news is fake.")
    else:
        st.warning("Please enter some text or upload an image.")

if st.button("Check Facts"):
    if user_input:
        with st.spinner("Fact-checking..."):
            result = user_friendly_fact_check(user_input, threshold)
            st.markdown(f"**Fact-Check Result:** {result['Fact-Check Result']}")
            st.markdown(f"**Confidence Level:** {result['Confidence Level']}")
            st.markdown(f"**Similarity Score:** {result['Similarity Score']}")
            if result['Entity URL'] != "No URL available":
                st.markdown(f"[More Info]({result['Entity URL']})")
    else:
        st.warning("Please enter some text or upload an image.")
        
with st.expander("ℹ️ How to Use"):
    st.write("""
        - Enter the text or upload image you want to verify.
        - Click the appropriate button to check for misinformation or fact-check.
        - In Advanced options in left panel set the threshold of similarity coefficient for fact checking.
    """)

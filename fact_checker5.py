import requests
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the sentence transformer for similarity checks
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Google Knowledge Graph API setup
API_KEY = 'AIzaSyC8SxCy92vwF0gpjZ2RF7uyolVcKaUJjMc'  # Replace with your actual API Key
service = build('kgsearch', 'v1', developerKey=API_KEY)

def get_google_kg_entity(query):
    # Query Google Knowledge Graph API
    response = service.entities().search(query=query, limit=1).execute()
    
    if 'itemListElement' in response:
        entity = response['itemListElement'][0]
        entity_name = entity['result']['name']
        entity_description = entity['result']['description'] if 'description' in entity['result'] else "No description available"
        entity_url = entity['result']['url'] if 'url' in entity['result'] else None
        return entity_name, entity_description, entity_url
    else:
        return None, None, None

def compute_similarity(input_text, page_content):
    input_embedding = embedder.encode([input_text])
    page_embedding = embedder.encode([page_content])
    return cosine_similarity(input_embedding, page_embedding)[0][0]

def fact_check(input_text):
    # Query Google Knowledge Graph with key terms
    key_terms = input_text.split()[:3]  # Extract first 3 words for better keyword matching
    best_match_score = 0
    best_match_details = None
    
    for term in key_terms:
        name, description, url = get_google_kg_entity(term)
        if name:
            similarity_score = compute_similarity(input_text, description)
            if similarity_score > best_match_score:
                best_match_score = similarity_score
                best_match_details = {"name": name, "description": description, "url": url}

    # Based on the similarity score, classify the result
    if best_match_score > 0.75:
        return {
            "result": "True",
            "details": f"High confidence match found on Google Knowledge Graph: {best_match_details['name']}",
            "similarity_score": best_match_score,
            "content_snippet": best_match_details['description'][:300] + "..."
        }
    elif best_match_score > 0.5:
        return {
            "result": "Likely True",
            "details": f"Moderate confidence match found on Google Knowledge Graph: {best_match_details['name']}. Verify further.",
            "similarity_score": best_match_score
        }
    else:
        return {
            "result": "False",
            "details": "No strong match found. Likely misinformation.",
            "similarity_score": best_match_score
        }

# Example usage
input_text = "Manish is the Prime Minister of India."
output = fact_check(input_text)
print(output)

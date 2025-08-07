import requests
import os

# Load your Bing API key from environment variable or hardcode (not recommended)
BING_API_KEY = os.getenv("BING_API_KEY")  # Or: BING_API_KEY = "your_api_key_here"
BING_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

def bing_search(query, count=3):
    if not BING_API_KEY:
        raise ValueError("Bing API Key not set in environment variable BING_API_KEY")

    headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
    params = {"q": query, "count": count}

    response = requests.get(BING_ENDPOINT, headers=headers, params=params)

    if response.status_code != 200:
        raise Exception(f"Bing Search API failed: {response.status_code} - {response.text}")

    results = response.json()
    web_pages = results.get("webPages", {}).get("value", [])
    
    # Format as plain text
    formatted_results = []
    for page in web_pages:
        title = page.get("name")
        url = page.get("url")
        snippet = page.get("snippet")
        formatted_results.append(f"**{title}**\n{url}\n{snippet}")
    
    return "\n\n".join(formatted_results)

from config.tavily import TAVILY_API_KEY
from tavily import TavilyClient
import streamlit as st
import time

def perform_web_search(query, max_results=3, include_domains=None, exclude_domains=None):
    """
    Enhanced web search with better error handling and Streamlit integration
    
    Args:
        query: Search query string
        max_results: Maximum number of results (1-5)
        include_domains: List of domains to include
        exclude_domains: List of domains to exclude
        
    Returns:
        List of dicts with keys: title, url, content, score
        OR None if failed
    """
    if not TAVILY_API_KEY:
        st.error("Tavily API key not configured")
        return None

    # Validate inputs
    if not query or not isinstance(query, str):
        st.error("Invalid search query")
        return None
        
    max_results = max(1, min(5, int(max_results)))  # Clamp to 1-5

    try:
        with st.spinner(f"Searching web for: '{query}'..."):
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            
            # Retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = tavily.search(
                        query=query,
                        search_depth="advanced",
                        max_results=max_results,
                        include_answer=True,
                        include_raw_content=True,
                        include_domains=include_domains or [],
                        exclude_domains=exclude_domains or [],
                    )
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff

            # Process results with quality filtering
            clean_results = []
            for result in response.get('results', []):
                if not result.get('url') or not result.get('content'):
                    continue
                    
                clean_results.append({
                    'title': result.get('title', 'No title available'),
                    'url': result['url'],
                    'content': result.get('content', ''),
                    'score': result.get('score', 0),
                    'last_updated': result.get('last_updated'),
                })
            
            # Sort by relevance score
            clean_results.sort(key=lambda x: x['score'], reverse=True)
            
            return clean_results[:max_results]

    except Exception as e:
        st.error(f"Web search failed: {str(e)}")
        with st.expander("Technical Details"):
            st.write(f"Query: {query}")
            st.write(f"Error type: {type(e).__name__}")
        return None
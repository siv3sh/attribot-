# import os
# import sys
# from langchain_groq import ChatGroq
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# def get_chatgroq_model():
#     """Initialize and return the Groq chat model"""
#     try:
#         # Initialize the Groq chat model with the API key
#         groq_model = ChatGroq(
#             api_key="",
#             model="",
#         )
#         return groq_model
#     except Exception as e:
#         raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")


import google.generativeai as genai
from config.config import GEMINI_API_KEY

def get_gemini_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-pro")
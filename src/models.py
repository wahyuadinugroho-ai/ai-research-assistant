from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_genai import HarmCategory, HarmBlockThreshold
from src.config import EMBEDDING_MODEL, GOOGLE_API_KEY, LLM_MODEL, TEMPERATURE

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


def get_llm(api_key=None):
    key = api_key or GOOGLE_API_KEY
    if not key:
        raise ValueError("Google API Key is missing.")

    kwargs = {
        "model": LLM_MODEL,
        "google_api_key": key,
        "safety_settings": safety_settings,
    }
    
    # gemini-3.5-flash-lite uses fixed sampling defaults, so temperature will be ignored and throw a warning
    if "flash-lite" not in LLM_MODEL.lower():
        kwargs["temperature"] = TEMPERATURE

    return ChatGoogleGenerativeAI(**kwargs)


def get_embeddings(api_key=None):
    key = api_key or GOOGLE_API_KEY
    if not key:
        raise ValueError("Google API Key is missing.")

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=key,
    )

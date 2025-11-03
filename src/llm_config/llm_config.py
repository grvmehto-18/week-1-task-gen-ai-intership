"""
LLM and Embeddings Configuration
This module configures the language model and text embeddings used in the chatbot.
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_llm(model_name: str = "gpt-4o-mini", base_url: str = None) -> ChatOpenAI:
    """
    Initializes and returns the ChatOpenAI model.

    Args:
        model_name (str): The name of the OpenAI model to use.
        base_url (str, optional): The base URL for the API. Defaults to None.

    Returns:
        ChatOpenAI: An instance of the ChatOpenAI model.
    """
    return ChatOpenAI(model=model_name, base_url=base_url)

def get_embeddings(model_name: str = "text-embedding-3-large", base_url: str = None) -> OpenAIEmbeddings:
    """
    Initializes and returns the OpenAIEmbeddings model.

    Args:
        model_name (str): The name of the OpenAI embedding model to use.
        base_url (str, optional): The base URL for the API. Defaults to None.

    Returns:
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings model.
    """
    return OpenAIEmbeddings(model=model_name, openai_api_base=base_url)

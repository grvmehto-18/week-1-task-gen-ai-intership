"""
Vector Store Service
This module handles the creation of the FAISS vector store.
"""
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def create_vector_store(dataframe: pd.DataFrame, embeddings: OpenAIEmbeddings) -> FAISS:
    """
    Creates a FAISS vector store from a pandas DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        embeddings (OpenAIEmbeddings): The embeddings model to use.

    Returns:
        FAISS: A FAISS vector store instance.
    """
    print("Creating vector store...")

    # Clean up column names
    dataframe = dataframe.rename(columns={
        'Drive_Configuration': 'drive_config',
        'Germany_price_before_incentives': 'price_de'
    })

    # Extract 'make' from 'title'
    if 'title' in dataframe.columns:
        dataframe['make'] = dataframe['title'].apply(lambda x: x.split(' ')[0])

    # Combine columns into richer text content
    dataframe['doc_content'] = dataframe.apply(
        lambda row: f"The {row['make']} {row['model']} is a {row['drive_config']} drive car priced in Germany at EUR {row['price_de']}.",
        axis=1
    )

    loader = DataFrameLoader(dataframe, page_content_column="doc_content")
    documents = loader.load()

    # Split docs
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Vector store
    db = FAISS.from_documents(texts, embeddings)
    # print("Vector store created successfully.")
    return db


import os
import pandas as pd
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import CharacterTextSplitter
from typing import Dict

class ChatbotService:
    """
    Service for the LangChain chatbot.
    Handles document loading, vector store creation, and question answering.
    """

    def __init__(self, dataframe: pd.DataFrame, openai_api_key: str):
        """
        Initializes the ChatbotService.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be used as a knowledge base.
            openai_api_key (str): The OpenAI API key.
        """
        if not openai_api_key:
            raise ValueError("OpenAI API key is required for the chatbot service.")
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.dataframe = dataframe
        self.qa_chain = self._create_qa_chain()

    def _create_qa_chain(self):
        """
        Creates the RetrievalQA chain.

        Returns:
            A RetrievalQA chain instance.
        """
        print("Creating chatbot QA chain...")
        
        # To make the document more informative, we'll combine some columns.
        # This creates a more descriptive text for each row.
        self.dataframe['doc_content'] = self.dataframe.apply(lambda row: f"The {row['make']} {row['model']} is a {row['drive_config']} drive with a range of {row['range']} km and a price in Germany of €{row['price_de']}.", axis=1)

        loader = DataFrameLoader(self.dataframe, page_content_column="doc_content")
        documents = loader.load()

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Create a FAISS vector store from the documents
        db = FAISS.from_documents(texts, embeddings)

        # Create a retriever interface
        retriever = db.as_retriever(search_kwargs={'k': 2})

        # Create the QA chain
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        print("Chatbot QA chain created successfully.")
        return qa

    def ask_question(self, query: str) -> Dict:
        """
        Asks a question to the chatbot.

        Args:
            query (str): The question to ask.

        Returns:
            Dict: The response from the QA chain.
        """
        print(f"Asking question: {query}")
        return self.qa_chain({"query": query})

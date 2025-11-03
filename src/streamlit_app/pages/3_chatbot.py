
import streamlit as st
import sys
import os
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Load environment variables from .env file
load_dotenv()

from src.services.data_service import DataService
from src.chatbot.chatbot_service import ChatbotService

st.set_page_config(page_title="EV Chatbot", page_icon="🤖")

st.title("EV Chatbot")

st.markdown("""
Ask questions about the EV dataset. The chatbot will do its best to answer them based on the available data.
""")

# --- API Key Handling ---
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key or openai_api_key == "YOUR_API_KEY_HERE":
    st.warning("Please add your OpenAI API key to the .env file in the project root.")
    st.stop()

@st.cache_resource
def initialize_chatbot(api_key):
    """
    Initializes the chatbot service.
    Cache this to avoid re-initializing on every interaction.
    """
    try:
        data_service = DataService(file_path='src/datasets/ev_raw_data.csv')
        df = data_service.get_dataframe_for_eda()
        return ChatbotService(dataframe=df, openai_api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return None

chatbot_service = initialize_chatbot(openai_api_key)

if chatbot_service is None:
    st.stop()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chatbot_service.ask_question(prompt)
                result = response.get('result', "I couldn't find an answer.")
                st.markdown(result)
                
                # Optionally display source documents
                with st.expander("See sources"):
                    st.write(response.get('source_documents', "No sources found."))
                
                st.session_state.messages.append({"role": "assistant", "content": result})

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

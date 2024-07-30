import streamlit as st
from streamlit_chat import message
import os
import tempfile
import json
from datetime import datetime, timedelta
from populate_database import load_documents, split_documents, add_to_chroma, clear_database
from rag_query import query_rag

# Paths for cloud deployment
CHROMA_PATH = "chroma"  # Use a relative path for cloud deployment
MEMORY_FILE = "chat_memory.json"
MEMORY_DURATION_DAYS = 3

# Initialize the Streamlit application
st.set_page_config(
    page_title="Conversational RAG System",
    page_icon=":robot:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def update_database(uploaded_file_path):
    global DATA_PATH
    original_data_path = DATA_PATH
    DATA_PATH = os.path.dirname(uploaded_file_path)
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    DATA_PATH = original_data_path

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as file:
            try:
                memory_data = json.load(file)
            except json.JSONDecodeError:
                return []
        timestamp = memory_data.get("timestamp")
        memory = memory_data.get("memory")
        if datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S") + timedelta(days=MEMORY_DURATION_DAYS) > datetime.now():
            return memory
    return []

def save_memory(memory):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(MEMORY_FILE, "w") as file:
        json.dump({"timestamp": timestamp, "memory": memory}, file)

def clear_memory():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)

# Function to handle user input
def handle_input():
    user_input = st.session_state['input']
    if user_input:
        casual_greetings = ["hello", "hi", "hey", "how are you", "good morning", "good afternoon", "good evening", "hola", "ohio", "wassup", "heya"]
        if user_input.lower() in casual_greetings:
            response = "Hello! How can I assist you today?"
        else:
            response = query_rag(user_input)
        st.session_state['history'].append({"user": user_input, "bot": response})
        memory = st.session_state['history']
        save_memory(memory)
        st.session_state['input'] = ''  # Clear input box after submitting

# Load chat history
if 'history' not in st.session_state:
    st.session_state['history'] = load_memory()

# Sidebar for document upload and restart chat
with st.sidebar:
    st.title("Pannel")

    if st.button("Restart Chat"):
        st.session_state['history'] = []
        clear_memory()
        st.experimental_set_query_params()  # Refresh the app

    with st.expander("Upload New Pdf Document", expanded=False):
        uploaded_file = st.file_uploader("Choose a file", type=["pdf"])  # Adjust file types as necessary
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            with st.spinner("Processing and updating database..."):
                update_database(temp_file_path)
                st.success("Database updated successfully!")
            os.remove(temp_file_path)

# Main chat interface
st.title("SmartQuery AI")

# Persistent text input at the top
st.text_input("You:", key="input", on_change=handle_input)

# Display chat history
for idx, chat in reversed(list(enumerate(st.session_state['history']))):
    message(chat['user'], is_user=True, key=f"user_{idx}")
    message(chat['bot'], is_user=False, key=f"bot_{idx}")

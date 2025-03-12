import streamlit as st
import os
import time
from datetime import datetime
import subprocess

# Set up the directory to save uploaded files
UPLOAD_DIR = "data/input/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to handle file upload and processing
def process_files(uploaded_files):
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.write(f"Processed file: {uploaded_file.name}")
    time.sleep(2)  # Simulate processing time

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for line in process.stdout:
        print(line.decode('utf-8').strip())

    stderr_output = process.stderr.read().decode('utf-8')
    if stderr_output:
        print(f"Error running command: {command}\n{stderr_output}")


# Function to simulate a chat response
def get_chat_response(user_input):
    return f"Current datetime: {datetime.now()}"

# Function to display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Function to handle user input and generate responses
def handle_user_input(prompt):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate a response
    response = get_chat_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload", "Assistant"])

    if page == "Upload":
        st.title("Upload Data")
        # File uploader
        uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

        if uploaded_files:
            st.write("Files uploaded:")
            for uploaded_file in uploaded_files:
                st.write(uploaded_file.name)

            # Submit button
            if st.button("Submit"):
                with st.spinner("Processing files please wait..."):
                    process_files(uploaded_files)
                    casemine_command = f'python -m chunk_folder'
                    run_command(casemine_command)

                st.success("Files processed and saved successfully!")

    elif page == "Assistant":
        st.title("Assistant")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        display_chat_messages()

        # Accept user input
        if prompt := st.chat_input("How can I help you"):
            handle_user_input(prompt)

if __name__ == "__main__":
    main()
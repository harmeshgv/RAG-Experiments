import streamlit as st
import time
import PyPDF2
from groq import Groq
import io  

# Set page config at the very beginning
st.set_page_config(layout="wide")

@st.cache_resource
def initialize_groq_client(api_key):
    """Initialize Groq client with API key."""
    return Groq(api_key=api_key)

client = initialize_groq_client(api_key="YOUR-API-KEY")

# Initialize conversation history for chat
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [
        {
            "role": "system",
            "content": "You are a data analyst, data engineer, and business analyst."
        }
    ]

# Extract text from PDF
def extract_pdf_text(uploaded_file):
    """Extract text from the PDF file."""
    file_bytes = uploaded_file.read()  # Read the file as bytes
    file_like_object = io.BytesIO(file_bytes)  # Create a file-like object from bytes
    reader = PyPDF2.PdfReader(file_like_object)  # Read the PDF from the file-like object
    text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Get response from Groq
def get_response(user_query, context):
    """Get a response from Groq's model with retry logic and improved error handling."""
    st.session_state.conversation_history.append({
        "role": "user",
        "content": f"{context}\n\n{user_query}"
    })

    conversation_history = st.session_state.conversation_history[-10:]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=conversation_history,
                model="Llama-3.3-70b-Versatile",
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )
            assistant_response = response.choices[0].message.content
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            return assistant_response
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = 2 ** attempt
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            elif "invalid api key" in str(e).lower():
                st.error("Invalid API key. Please check your API key and try again.")
                return None
            else:
                st.error(f"Error: {e}")
                return None

# Streamlit UI
st.title("Stat IQ - PDF Q&A System")
st.write("Upload your PDF file and ask questions related to its content.")

uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_pdf is not None:
    # Extract text from the uploaded PDF
    pdf_text = extract_pdf_text(uploaded_pdf)

    # Display PDF content (optional for debugging)
    st.subheader("Extracted PDF Text:")
    st.write(pdf_text[:1500])  # Displaying the first 1500 characters of the extracted text

    # Get user question
    question = st.text_input("Ask a question related to the PDF:")

    if st.button("Submit Question"):
        if question:
            # Use the extracted PDF text as context for the question
            context = pdf_text
            with st.spinner('Generating response...'):
                response = get_response(question, context)
                st.write("Response:", response)
        else:
            st.error("Please enter a question.")

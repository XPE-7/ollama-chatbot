import os
import ssl

# CRITICAL: Fix SSL issues BEFORE importing langchain_ollama
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ.pop("CURL_CA_BUNDLE", None)

# Additional SSL bypass
ssl._create_default_https_context = ssl._create_unverified_context

# Set environment variables to disable SSL verification
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_INSECURE"] = "1"

# Now import the packages after SSL fixes
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

load_dotenv()

# LangChain tracing (optional)
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

## Streamlit framework
st.title('🦙 Langchain Demo With LLAMA2 API')
st.write("Ask me anything and I'll respond using LLAMA2!")

input_text = st.text_input("Search the topic you want", placeholder="e.g., What is machine learning?")

# Initialize LLM with error handling
@st.cache_resource
def initialize_llm():
    try:
        llm = OllamaLLM(
            model="llama2", 
            base_url="http://localhost:11434",
            # Additional parameters to help with connection
            timeout=30
        )
        # Test the connection with a simple prompt
        test_response = llm.invoke("Hi")
        st.success("✅ Connected to Ollama successfully!")
        return llm
    except Exception as e:
        st.error(f"❌ Failed to connect to Ollama: {str(e)}")
        st.info("Make sure Ollama is running with: `ollama serve`")
        st.info("And that llama2 model is available: `ollama pull llama2`")
        return None

# Initialize LLM
llm = initialize_llm()

if llm:
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    if input_text:
        with st.spinner("🤔 Thinking..."):
            try:
                response = chain.invoke({"question": input_text})
                st.write("**Response:**")
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.info("Make sure Ollama server is running and llama2 model is available.")

# Sidebar with instructions
with st.sidebar:
    st.header("📋 Setup Instructions")
    st.write("1. Start Ollama server:")
    st.code("ollama serve")
    st.write("2. Pull LLAMA2 model:")
    st.code("ollama pull llama2")
    st.write("3. Verify model:")
    st.code("ollama list")
    
    st.header("🔧 Current Status")
    if llm:
        st.success("Ollama: Connected ✅")
    else:
        st.error("Ollama: Not Connected ❌")
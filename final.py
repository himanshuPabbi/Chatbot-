import streamlit as st
import os
import pdfplumber
import time
import uuid

import gspread
from google.oauth2.service_account import Credentials

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_tavily import TavilySearch

# -----------------------------
# Initialize API keys from secrets
# -----------------------------
try:
    groq_api_key = st.secrets["GROQ"]["API_KEY"]
    tavily_api_key = st.secrets["TAVILY"]["API_KEY"]
    os.environ["TAVILY_API_KEY"] = tavily_api_key
except KeyError:
    st.error("API keys not found in .streamlit/secrets.toml. Please add them correctly.")
    st.stop()

# -----------------------------
# Google Sheets Client Setup
# -----------------------------
def get_gsheet_client():
    credentials_dict = st.secrets["gcp_service_account"]
    sheet_id = credentials_dict["sheet_id"]
    
    credentials = Credentials.from_service_account_info(
        credentials_dict,
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(credentials)
    return client, sheet_id

def log_performance(query_id, mode, query, response_text, duration, status):
    try:
        client, sheet_id = get_gsheet_client()
        sheet = client.open_by_key(sheet_id).sheet1

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())  # UTC time
        row = [
            query_id,
            timestamp,
            mode,
            query,
            len(response_text),
            int(duration * 1000),
            status
        ]
        sheet.append_row(row)
    except Exception as e:
        st.error(f"Error logging data to Google Sheets: {e}")

# -----------------------------
# Define LLMs
# -----------------------------
llm_for_qa = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
llm_for_general = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
llm_for_math = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# -----------------------------
# PDF Processing Functions
# -----------------------------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def trim_text(text, max_tokens=3000):
    tokens = text.split()
    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens])
    return text

def process_pdf_with_langchain(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.create_documents([pdf_text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm_for_qa, retriever=retriever, return_source_documents=True)
    return qa_chain

# -----------------------------
# Research & Math Agents
# -----------------------------
search = TavilySearch(max_results=10)
tools = [
    Tool(
        name="Tavily Search",
        description="A search tool to get the latest information from the web.",
        func=search.invoke
    )
]

research_agent = initialize_agent(
    tools,
    llm_for_general,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def handle_research(query):
    try:
        response = research_agent.run(query)
        return response
    except Exception as e:
        return f"An error occurred while running the search agent: {e}"

def handle_math(query):
    try:
        prompt_template = PromptTemplate.from_template(
            "You are a math tutor. Solve the following problem step-by-step and provide the final answer. Explain your reasoning clearly: {query}"
        )
        prompt = prompt_template.format(query=query)
        response = llm_for_math.invoke(prompt)
        return getattr(response, "content", response)
    except Exception as e:
        return f"An error occurred while solving the math problem: {e}"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("BrainyBot: Your Intelligent Assistant")
st.markdown("This app can answer questions about a PDF, perform general research, and solve math problems.")

mode = st.selectbox(
    "Choose a mode:",
    ("PDF Q&A", "Research & Latest Topics", "Solve Math Problem")
)

user_query = st.text_input("Enter your query:", "")

if user_query:
    query_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    if mode == "PDF Q&A":
        pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if pdf_file:
            pdf_path = "uploaded_pdf.pdf"
            try:
                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.read())
                
                pdf_text = extract_text_from_pdf(pdf_path)
                trimmed_pdf_text = trim_text(pdf_text)

                if trimmed_pdf_text:
                    qa_chain = process_pdf_with_langchain(trimmed_pdf_text)
                    response = qa_chain({"query": user_query})['result']
                    st.write("Chatbot:", response)
                    duration = time.perf_counter() - start_time
                    log_performance(query_id, "PDF Q&A", user_query, response, duration, "Success")
                else:
                    st.error("Could not extract text from the PDF.")
                    duration = time.perf_counter() - start_time
                    log_performance(query_id, "PDF Q&A", user_query, "Extraction Failed", duration, "Error")
            finally:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

    elif mode == "Research & Latest Topics":
        response = handle_research(user_query)
        duration = time.perf_counter() - start_time
        st.write("Research Summary:", response)
        log_performance(query_id, "Research", user_query, response, duration, "Success")

    elif mode == "Solve Math Problem":
        response = handle_math(user_query)
        duration = time.perf_counter() - start_time
        st.write("Solution:", response)
        log_performance(query_id, "Math", user_query, response, duration, "Success")

if st.button("Clear App"):
    st.session_state.clear()
    st.experimental_rerun()

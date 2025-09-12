import streamlit as st
import os
import pdfplumber
import time
import json
import uuid  # For unique query IDs

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

# --- Updated: Initialize API keys using Streamlit secrets ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    tavily_api_key = st.secrets["TAVILY_API_KEY"]
    os.environ["TAVILY_API_KEY"] = tavily_api_key
except KeyError:
    st.error("API keys not found in .streamlit/secrets.toml. Please create the file and add your keys.")
    st.stop()

# Define models
llm_for_qa = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
llm_for_general = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
llm_for_math = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# Performance logging
def log_performance(query_id, mode, query, response_text, duration, status):
    log_file = "performance_log.json"

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump([], f)

    try:
        with open(log_file, "r+") as f:
            content = f.read().strip()
            if not content:
                data = []
            else:
                data = json.loads(content)

            log_entry = {
                "query_id": query_id,
                "timestamp": time.time(),
                "mode": mode,
                "query": query,
                "response_length": len(response_text),
                "duration_ms": int(duration * 1000),
                "status": status
            }
            data.append(log_entry)

            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=4)

    except json.JSONDecodeError:
        with open(log_file, "w") as f:
            log_entry = {
                "query_id": query_id,
                "timestamp": time.time(),
                "mode": mode,
                "query": query,
                "response_length": len(response_text),
                "duration_ms": int(duration * 1000),
                "status": status
            }
            json.dump([log_entry], f, indent=4)

# PDF processing functions
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

# Research & Math functionality
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
    st.info("Searching for the latest information using Tavily Search...")
    try:
        response = research_agent.run(query)
        return response
    except Exception as e:
        return f"An error occurred while running the search agent: {e}"

def handle_math(query):
    st.info("Solving the math problem...")
    prompt_template = PromptTemplate.from_template(
        "You are a math tutor. Solve the following problem step-by-step and provide the final answer. Explain your reasoning clearly: {query}"
    )
    prompt = prompt_template.format(query=query)
    response = llm_for_math.invoke(prompt)
    return response.content

# Streamlit UI
st.title("BrainyBot: Your Intelligent Assistant")
st.markdown("This app can answer questions about a PDF, perform general research, and solve math problems.")

mode = st.selectbox(
    "Choose a mode:",
    ("PDF Q&A", "Research & Latest Topics", "Solve Math Problem")
)

user_query = st.text_input("Enter your query:", "")

if user_query:
    query_id = str(uuid.uuid4())

    if mode == "PDF Q&A":
        pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if pdf_file:
            start_time = time.time()
            pdf_path = "uploaded_pdf.pdf"
            with open(pdf_path, "wb") as f:
                f.write(pdf_file.read())

            with st.spinner("Extracting text and setting up Q&A system..."):
                pdf_text = extract_text_from_pdf(pdf_path)
                trimmed_pdf_text = trim_text(pdf_text)

                if trimmed_pdf_text:
                    qa_chain = process_pdf_with_langchain(trimmed_pdf_text)

                    with st.spinner("Generating response..."):
                        try:
                            response = qa_chain({"query": user_query})['result']
                            duration = time.time() - start_time
                            st.success("Response generated!")
                            st.write("Chatbot:", response)
                            log_performance(query_id, "PDF Q&A", user_query, response, duration, "Success")
                        except Exception as e:
                            duration = time.time() - start_time
                            st.error(f"An error occurred: {e}")
                            log_performance(query_id, "PDF Q&A", user_query, str(e), duration, "Error")
                else:
                    duration = time.time() - start_time
                    st.error("Could not extract text from the PDF. Please try a different file.")
                    log_performance(query_id, "PDF Q&A", user_query, "Extraction Failed", duration, "Error")

            os.remove(pdf_path)

    elif mode == "Research & Latest Topics":
        start_time = time.time()
        with st.spinner("Searching and summarizing information..."):
            try:
                response = handle_research(user_query)
                duration = time.time() - start_time
                st.success("Response generated!")
                st.write("Research Summary:", response)
                log_performance(query_id, "Research", user_query, response, duration, "Success")
            except Exception as e:
                duration = time.time() - start_time
                st.error(f"An error occurred: {e}")
                log_performance(query_id, "Research", user_query, str(e), duration, "Error")

    elif mode == "Solve Math Problem":
        start_time = time.time()
        with st.spinner("Solving the problem..."):
            try:
                response = handle_math(user_query)
                duration = time.time() - start_time
                st.success("Solution found!")
                st.write("Solution:", response)
                log_performance(query_id, "Math", user_query, response, duration, "Success")
            except Exception as e:
                duration = time.time() - start_time
                st.error(f"An error occurred: {e}")
                log_performance(query_id, "Math", user_query, str(e), duration, "Error")

if st.button("Clear App"):
    st.experimental_rerun()

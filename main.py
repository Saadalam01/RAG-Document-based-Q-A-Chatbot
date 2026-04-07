#RAG conversational Q&A chatbot with pdf including chat history

import streamlit as st
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import io


import os
from dotenv import load_dotenv
load_dotenv()

# loading HF tokens
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# setup streamlit
st.title("RAG conversational Q&A chatbot with pdf including chat history")
st.write("Upload PDFs and chat with their content.")

# Input GROQ api key
api_key = st.text_input("Enter your GROQ API key:", type="password")

# Check if Groq api key is provided
if not api_key:
    st.warning("Please enter your GROQ API key to continue.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")

# chat interface
session_id = st.text_input("Session ID", value="Default")

# statefully manage chat history
if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# process uploaded files
documents = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            pdf_path = tmp.name
        
        # Load the PDF from the temporary path
        try:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
            st.success(f"'{uploaded_file.name}' uploaded and processed successfully!")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            # Clean up the temporary file
            os.remove(pdf_path)


if not documents:
    st.info("Upload at least one PDF to continue.")
    st.stop()

# Split and create document embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Create a history-aware retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Create a question-answering chain
qa_system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Combine them into a RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Function to get session history
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# Create a conversational RAG chain with message history
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Handle user input
user_input = st.text_input("Your question:")
if user_input:
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )
    st.write("Assistant:", response["answer"])

    # Display chat history for debugging/verification
    with st.expander("View Chat History"):
        # Safely access and display session history
        history = st.session_state.store.get(session_id)
        if history:
            st.write(history.messages)
        else:
            st.write("No history yet.")

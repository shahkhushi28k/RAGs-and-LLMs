import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import toml

config = toml.load('config.toml')  # Path to your TOML file
OPENAI_API_KEY = config['api_keys']['OPENAI_API_KEY']


# Load environment variables
load_dotenv()

# API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Fix for OpenMP runtime conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Cache the embedding creation to avoid repeated initialization
@st.cache_resource
def initialize_embeddings():
    return OpenAIEmbeddings(model="text-embedding-ada-002")

# Cache the conversational model to avoid reinitialization
@st.cache_resource
def initialize_chat_model():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# Cache the document processing workflow
@st.cache_resource
def process_document(file_path):
    # Load the document using PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = initialize_embeddings()
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    return retriever

# Cache the model initialization for summarization
@st.cache_resource
def load_summarizer():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    return tokenizer, model

# Preprocess input for summarization
def preprocess_input(text, tokenizer):
    return tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary using the summarizer model
def generate_summary(text, tokenizer, model):
    inputs = preprocess_input(text, tokenizer)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary.split(". ")

# Define the main functionality for Chat with Document
def Chat_with_document_using_RAG():
    st.title("Chat with Document using RAG")
    st.write("Chat with your uploaded documents!")

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file:
        st.success("Document uploaded successfully!")
        with st.spinner("Processing the document..."):
            # Save the uploaded file temporarily
            file_path = "uploaded_document.pdf"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Process the document and initialize retriever
            retriever = process_document(file_path)

        # Set up memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, max_memory_length=5)

        # Load the conversational model
        model = initialize_chat_model()

        # Create the retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=memory
        )

        st.write("You can now ask questions about your document!")
        user_input = st.text_input("Ask a question about the document:")
        if st.button("Get Answer"):
            if user_input.strip():
                with st.spinner("Generating answer..."):
                    # Generate the response using the chain
                    result = chain.invoke({"question": user_input, "chat_history": []})
                    st.write("**Answer:**", result["answer"])
            else:
                st.error("Please enter a question.")

# Define the Meeting Notes Summarizer functionality
def Meeting_Notes_Summarizer_using_LLMS():
    tokenizer, model = load_summarizer()

    st.title("Meeting Notes Summarizer")
    st.write("This web app summarizes your meeting notes into concise bullet points.")

    user_input = st.text_area("Enter Meeting Notes:")
    if st.button("Summarize Meeting Notes"):
        if user_input:
            st.write("### Original Meeting Notes:")
            st.write(user_input)

            summary_points = generate_summary(user_input, tokenizer, model)
            st.write("### Summarized Meeting Notes:")
            for point in summary_points:
                st.write(f"• {point}")
        else:
            st.write("Please enter meeting notes to summarize.")

# Main function to handle navigation
def main():
    st.sidebar.title("Choose a Functionality")
    options = ["Chat with Document using RAG", "Meeting Notes Summarizer using LLMs"]
    choice = st.sidebar.radio("Select an option:", options)

    if choice == "Chat with Document using RAG":
        Chat_with_document_using_RAG()
    elif choice == "Meeting Notes Summarizer using LLMs":
        Meeting_Notes_Summarizer_using_LLMS()

if __name__ == "__main__":
    main()

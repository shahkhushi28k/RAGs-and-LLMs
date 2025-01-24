 
# Document Chat & Meeting Notes Summarizer

This application provides two main functionalities:

1. **Chat with Document using Retrieval-Augmented Generation (RAG):**
   Upload a PDF document and ask questions about its content. The app processes the document, splits it into chunks, and uses OpenAI's embedding and conversational AI models to provide answers.

2. **Meeting Notes Summarizer using Large Language Models (LLMs):**
   Summarize meeting notes into concise bullet points using the T5 model from Hugging Face.

## Features

- Upload and chat with PDF documents.
- Generate concise summaries of meeting notes.
- Uses LangChain for document processing and conversational chains.
- Incorporates T5 from Hugging Face for text summarization.

## Requirements

- Python.
- Required dependencies are listed in `requirements.txt`.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/document-chat-summarizer.git
   cd document-chat-summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up OpenAI API key:
   - Create a `.env` file in the project directory.
   - Add the following line with your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Choose a functionality from the sidebar:
   - **Chat with Document using RAG**: Upload a PDF and ask questions.
   - **Meeting Notes Summarizer using LLMs**: Paste meeting notes to generate summaries.

## File Structure

- `main.py`: Main application file.
- `requirements.txt`: List of required dependencies.
- `.env`: File to store API keys (not included in the repo).

## Models Used

- **ChatGPT (GPT-3.5-turbo)**: Used for conversational retrieval chains.
- **T5 (small)**: Used for summarizing text.

"# RAGs-and-LLMs" 

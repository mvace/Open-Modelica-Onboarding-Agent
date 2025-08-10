# OpenModelica Onboarding Assistant

An AI-powered assistant for helping users learn and work with OpenModelica. This project uses LangChain, OpenAI, and Streamlit to create an interactive QA system that provides accurate, documented answers about OpenModelica.

**How it works:**  
The application ingests the *OpenModelica User’s Guide* PDF, splits it into overlapping text chunks, generates embeddings for each chunk using OpenAI’s embedding model, and stores them in a FAISS vector index. The Streamlit web UI lets the user enter natural language questions about OpenModelica. When a query is submitted, the system retrieves the most relevant chunks from the FAISS index, passes them along with the user’s question to a GPT-4-class model via LangChain, and uses a custom prompt to produce a concise, cited answer. The response is displayed in the UI with clickable links to the exact PDF pages for reference.

## Setup

1. Create a virtual environment:

    ```bash
    python -m venv .venv
    ```

1. Activate the virtual environment:

    For Linux/macOS:
    ```bash
    source .venv/bin/activate
    ```

    For Windows:
    ```bash
    .venv\Scripts\activate
    ```

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

1. Configure environment variables:

    ```bash
    cp .env.example .env
    # Edit .env with your OpenAI API key and other settings
    ```
### Optional:

You can skip this step if running the project with the provided `storage/faiss_openmodelica/` folder.
If you need to rebuild (e.g., new PDF version or missing index):

```bash
python -m ingest.build_index
```
    

## Usage

1. Start the Streamlit interface:

    ```bash
    streamlit run app.py
    ```

1. Open your browser and navigate to the provided URL (typically `http://localhost:8501`)

1. Ask questions about OpenModelica in the chat interface
from flask import Flask, request, render_template
import requests
from collections import OrderedDict
# Import other necessary modules and configurations from your script here
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os

app = Flask(__name__)

# You might need to adjust these imports based on your project structure
from example import search_documents, filter_documents, create_embeddings, store_documents, answer_with_langchain, Document
# Library imports
from collections import OrderedDict
import requests
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
# Lang Chain library imports
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
# from langchain.vectorstores import Chroma
# Configuration imports
from config import (
    SEARCH_SERVICE_ENDPOINT,
    SEARCH_SERVICE_KEY,
    SEARCH_SERVICE_API_VERSION,
    SEARCH_SERVICE_INDEX_NAME1,
    SEARCH_SERVICE_SEMANTIC_CONFIG_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
)

# Azure AI Search service header settings
HEADERS = {
    'Content-Type': 'application/json',
    'api-key': SEARCH_SERVICE_KEY
}
# Azure Blob Storage settings
connect_str = 'DefaultEndpointsProtocol=https;AccountName=hackathonteam57;AccountKey=QZntSMy9cUX9Vj/2C9HlEiba5QHdEqCNBdngkHeADAKN96CIw8jFydytbeGLkXrqkuhJBH1qTL5l+AStEGigoA==;EndpointSuffix=core.windows.net'
container_name = 'test'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)


def extract_text_from_document(file_stream):
    """
    Extracts text from a PDF document.
    
    Args:
    - file_stream: The binary stream of the PDF file.
    
    Returns:
    - Extracted text as a string.
    """
    # Load the PDF file from the file stream
    try:
        pdf_document = fitz.open(stream=file_stream, filetype="pdf")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # Load the current page
        text += page.get_text()  # Extract text from the page

    pdf_document.close()  # Close the document
    return text


def search_documents(question):
    """Search documents using Azure AI Search."""
    # Construct the Azure AI Search service access URL.
    url = (SEARCH_SERVICE_ENDPOINT + 'indexes/' +
                SEARCH_SERVICE_INDEX_NAME1 + '/docs')
    
    # Create a parameter dictionary.
    params = {
        'api-version': SEARCH_SERVICE_API_VERSION,
        'search': question,
        'select': '*',
        # '$top': 5, Extract the top 5 documents from your storage.
        '$top': 5,
        'queryLanguage': 'en-us',
        'queryType': 'semantic',
        'semanticConfiguration': SEARCH_SERVICE_SEMANTIC_CONFIG_NAME,
        '$count': 'true',
        'speller': 'lexicon',
        'answers': 'extractive|count-3',
        'captions': 'extractive|highlight-false'
        }
    # Make a GET request to the Azure AI Search service and store the response in a variable.
    resp = requests.get(url, headers=HEADERS, params=params)
    # Return the JSON response containing the search results.
    search_results = resp.json()

    return search_results
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'document' not in request.files:
        return 'No file part', 400
    file = request.files['document']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = secure_filename(file.filename)
        file_stream = file.read()  # Read the file content for processing

        # Extract text from the uploaded document
        document_text = extract_text_from_document(file_stream)

        if document_text is None:
            return 'Failed to extract text from the document.', 500
        summary = generate_summary(document_text)
        # Proceed with processing the extracted text...
        # (Create embeddings, store in vector store, etc.)
        trigger_indexer("blob-indexer")
        return render_template('upload.html', summary=summary),200
    else:
        return 'Invalid file type', 400

def filter_documents(search_results):
    """Filter documents with a reranker score above a certain threshold."""
    documents = OrderedDict()
    for result in search_results['value']:
        # The '@search.rerankerScore' range is 0 to 4.00, where a higher score indicates a stronger semantic match.
        if result['@search.rerankerScore'] > 0.2:
            documents[result['metadata_storage_path']] = {
                'chunks': result['pages'][:10],
                'captions': result['@search.captions'][:10],
                'score': result['@search.rerankerScore'],
                'file_name': result['metadata_storage_name']
            }

    return documents
def trigger_indexer(indexer_name):
    url = f"{SEARCH_SERVICE_ENDPOINT}/indexers/{indexer_name}/run?api-version={SEARCH_SERVICE_API_VERSION}"
    response = requests.post(url, headers=HEADERS)
    if response.status_code == 202:
        print("Indexer triggered successfully.")
    else:
        print("Failed to trigger indexer.", response.json())

def create_embeddings():
    """Create an embedding model."""
    embeddings = AzureOpenAIEmbeddings(
        openai_api_key = AZURE_OPENAI_KEY,
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
        openai_api_version = AZURE_OPENAI_API_VERSION,
        openai_api_type = 'azure',
        azure_deployment = 'text-embedding-ada-002',
        model = 'text-embedding-ada-002',
        chunk_size=1
    )
    return embeddings
def store_documents(docs, embeddings):
    """Create vector store and store documents in the vector store."""
    return FAISS.from_documents(docs, embeddings)
 
def answer_with_langchain(vector_store, question):
    """Search for documents related to your question from the vector store
    and answer question with search result using Lang Chain."""
    # Add a chat service.
    llm = AzureChatOpenAI(
        openai_api_key = AZURE_OPENAI_KEY,
        azure_endpoint = AZURE_OPENAI_ENDPOINT,
        openai_api_version= AZURE_OPENAI_API_VERSION,
        azure_deployment = 'gpt-35-turbo',
        temperature=0.0,
        max_tokens=500
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    answer = chain.invoke({'question': question})

    return answer
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        question = request.form.get('question')
        if question:
            answer, references = process_question(question)
            return render_template('result.html', question=question, answer=answer, references=references)
    # Render home page with form if not POST request or no question provided
    return render_template('index.html')
def generate_summary(text, num_sentences=3):
    """Generate a simple summary by taking the first few sentences."""
    sentences = text.split('. ')
    summary = '. '.join(sentences[:num_sentences])
    return summary + '.'

def process_question(question):

    search_results = search_documents(question)
    documents = filter_documents(search_results)
    docs = []
    for key, value in documents.items():
        for page in value['chunks']:
            docs.append(Document(page_content=page, metadata={"source": value["file_name"]}))
    embeddings = create_embeddings()
    vector_store = store_documents(docs, embeddings)
    result = answer_with_langchain(vector_store, question)
    answer = result['answer']
    references = result['sources'].replace(",", "\n")
    return answer, references

if __name__ == '__main__':
    app.run(debug=True)

# Library imports
from collections import OrderedDict
import requests

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

def filter_documents(search_results):
    """Filter documents with a reranker score above a certain threshold."""
    documents = OrderedDict()
    for result in search_results['value']:
        # The '@search.rerankerScore' range is 0 to 4.00, where a higher score indicates a stronger semantic match.
        if result['@search.rerankerScore'] > 0.8:
            documents[result['metadata_storage_path']] = {
                'chunks': result['pages'][:10],
                'captions': result['@search.captions'][:10],
                'score': result['@search.rerankerScore'],
                'file_name': result['metadata_storage_name']
            }

    return documents

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

def main():
    QUESTION = 'explain Open for Business Fund'

    # Search for documents with Azure AI Search.

    search_results = search_documents(QUESTION)
    print(search_results)

    documents = filter_documents(search_results)

    print('Total Documents Found: {}, Top Documents: {}'.format(
        search_results['@odata.count'], len(search_results['value'])))
   
    print(documents.items())
    print("\n\n\n\n\n\n\n")
       # 'chunks' is the value that corresponds to the Pages field that you set up in the AI Search service.
    docs = []
    for key,value in documents.items():
        for page in value['chunks']:
            docs.append(Document(page_content = page,
                                metadata={"source": value["file_name"]}))

    # Answer your question using Lang Chain.
  
    embeddings = create_embeddings()
    print(embeddings)
    # raw_docs = ["Document 1 content", "Document 2 content", "Document 3 content"]
    # docs = [Document(content) for content in raw_docs]
    vector_store = store_documents(docs, embeddings)

    result = answer_with_langchain(vector_store, QUESTION)

    print('Question: ', QUESTION)
    print('Answer: ', result['answer'])
    print('Reference: ', result['sources'].replace(",","\n"))

# Execute the main function.
if __name__ == "__main__":
    main()

**Purpose:** Briefly describe the purpose of the Flask application, including its role in handling document uploads, text extraction, making reports and Q&A functionalities.

**Scope:** Ask Questions with already existing Documents and Generate the Reports for new documents .


**Components Description :**
- Azure blob for storing the documents 
- Azure AI Cognitive Search for indexing the documents
- Azure OpenAI service is used for how the application integrates with Azure services, including required APIs, keys, and configurations.
- Flask Web Server: Handling HTTP requests and responses.

**Models Used **:
text-embedding-ada-002 for embedding and generating vector data
gpt-3.5 for Q&A

**Architecture Design**
Please refer to hackathon.png


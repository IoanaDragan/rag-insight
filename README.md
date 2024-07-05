# RAG Insight

A Streamlit RAG Chatbot for querying PDF documents using a structure-aware chunking approach.


## Features

### PDF Indexer


The PDF Indexer uses the [LLMSherpa](https://github.com/nlmatics/llmsherpa) API internally for parsing the PDF document. The main sections thus obtained are split recursively using a chunk size of 2048 characters, into subsections that fit within this limit. The hierarchical structure of the document is maintained by returning entire sections rather than arbitrary slices of text. 
The resulting text chunks are used for building a [LlamaIndex](https://github.com/run-llama/llama_index) query engine, on top of an in-memory VectorStoreIndex.

### Streamlit Chatbot

The Streamlit Chatbot allows users to:

1. Input an OpenAI API key
2. Upload a PDF document
3. Ask questions about the document

## Installation

### Set up LLMSherpa API

1. Install the nlm-ingestor server:  
   Follow the instructions at https://github.com/nlmatics/nlm-ingestor

2. The local `llmsherpa_url` will be: `http://localhost:5001/api/parseDocument?renderFormat=all`


### Deploy the Chatbot

1. Clone the repository
```
git clone https://github.com/IoanaDragan/rag-insight
cd rag-insight
```

2. Install dependencies
```
pip install -r requirements.txt
```

3. (Optional) Set up environment variables:
Create a `.env` file and add your OpenAI API key:  
OPENAI_API_KEY=your_api_key_here

4. Launch the Streamlit server
```
streamlit run rag-chatbot.py
```

5. Access the app at `http://localhost:8501` in your browser


## Customization


The `index_pdf` method of the PDFIndexer accepts the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `chunk_size` | Size of document chunks | 2048 |
| `first_n_chunks` | Number of chunks to index (for testing) | None (all) |
| `add_summary` | Add chunk summaries as metadata | False |
| `retrieve_top_k` | Number of similar documents to retrieve per query | 2 |
| `similarity_threshold` | Minimum similarity score for retrieved documents | 0.8 |

 ## License

 Distributed under the MIT License. See [LICENSE](https://github.com/IoanaDragan/rag-insight/LICENSE.txt) for more information.


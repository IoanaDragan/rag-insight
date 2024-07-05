import os
from dotenv import load_dotenv
import streamlit as st

from pdf_indexer import PDFIndexer

# load variables from .env file
load_dotenv()

def similarity_top_k_changed():
    if st.session_state.is_document_indexed:
        pdf_indexer.update_similarity_top_k(st.session_state.similarity_top_k)

def similarity_threshold_changed():
    if st.session_state.is_document_indexed:
        pdf_indexer.update_similarity_threshold(st.session_state.similarity_threshold)

with st.sidebar:
    st.markdown("## How to use\n"
                "1. Enter your OpenAI API key below\n"
                "2. Upload a pdf document\n"
                "3. Ask questions about the document")

    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    st.divider()
    st.header("Settings")
    similarity_top_k = st.slider("Retrieval top-k", 1, 5, 3, key="similarity_top_k", on_change=similarity_top_k_changed,
                                 help="The number of top-k retrieved document chunks to consider for the query engine. ")
    
    similarity_threshold = st.slider("Retrieval similarity score", 0.5, 1.0, 0.8, key="similarity_threshold", on_change=similarity_threshold_changed,
                                     help = "Lowers or increases the similarity threshold for retrieved document chunks. "
                                     "A lower score will make the retriever more permissive, selecting text with a lower similarity to the query.")

    st.divider()
    st.markdown("## How does it work? \n"
                "When you upload a PDF document, it is divided into chunks of text using LLM Sherpa's advanced chunking algorithm. "
                "This algorithm aims to maintain the natural document's section-level structure, keeping related text together. \n\n"
                "The chunks of text are then embedded and stored in an in-memory LlamaIndex vector store index, enabling semantic search and retrieval. \n\n"
                "When you ask a question, the query engine retrieves the most relevant chunks of text from the document, based on the similarity of the query to the text. \n\n"
                "Text embedding and query answering are powered by OpenAI models.")

st.title("💬 RAG Insight")
st.caption("🚀 A Streamlit RAG chatbot powered by OpenAI, LLamaIndex and LLM Sherpa")


uploaded_file = st.file_uploader("Upload a PDF document", 
                                 type=("pdf"), 
                                 disabled=not openai_api_key and not os.environ.get("OPENAI_API_KEY"))

@st.cache_resource
def index_document(file_name, doc_content):
    pdf_indexer = PDFIndexer(openai_api_key=openai_api_key)
    pdf_indexer.index_pdf(path_or_url=file_name, content=doc_content, add_summary=True, 
                          retrieve_top_k=similarity_top_k, similarity_threshold=similarity_threshold)
    return pdf_indexer

if uploaded_file:
    file_name = uploaded_file.name
    doc_content = uploaded_file.getvalue()
    
    with st.status("Indexing document..."):
        pdf_indexer = index_document(file_name, doc_content) 
        st.session_state['is_document_indexed'] = True
        query_engine = pdf_indexer.get_query_engine()

if "is_document_indexed" not in st.session_state or not uploaded_file:
    st.session_state["is_document_indexed"] = False
    
if "messages" not in st.session_state or not uploaded_file:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


user_prompt = st.chat_input(
    "Ask something about the document",
    disabled=(not uploaded_file or st.session_state["is_document_indexed"] is False)
)

DEFAULT_EMPTY_RESPONSE = "I'm sorry, I can't find the information in the provided document."

def retrieve_context(query):
    retrieved_nodes = query_engine.retrieve(query)
    return '  \n\n'.join([f'Node score: {node.score}  \n Node text: {node.text}' for node in retrieved_nodes])

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").write(user_prompt)

    with st.spinner("..."):
        bot_response = query_engine.query(user_prompt).response
        bot_response = DEFAULT_EMPTY_RESPONSE if bot_response == "Empty Response" else bot_response

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        st.chat_message("assistant").write(bot_response)

        with st.expander("See context"):
            st.write(retrieve_context(user_prompt))
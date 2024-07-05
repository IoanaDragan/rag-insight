import os

from llmsherpa.readers import LayoutPDFReader
from llmsherpa.readers import Section

from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.extractors import SummaryExtractor
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.ingestion import IngestionPipeline

CHUNK_SIZE = 2048

class PDFIndexer:
    LLMSHERPA_URL = "http://localhost:5001/api/parseDocument?renderFormat=all"

    def __init__(self, llmsherpa_url=LLMSHERPA_URL, openai_api_key=None):
        self.pdf_reader = LayoutPDFReader(llmsherpa_url)

        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self._index = None
        self._response_synthesizer = None
        self._query_engine = None
        self._retriever = None


    def _split_section_to_text(self, section, chunk_size=CHUNK_SIZE):
        sub_sections_as_text = []

        section_text = ''
        for child in section.children:
            child_text = child.to_text(include_children=True, recurse=True)

            # recursively split section if it is too large, otherwise append it to the current section
            if isinstance(child, Section):
                if section_text:
                    sub_sections_as_text.append(section.parent_text() + "\n" + section.title + "\n" + section_text)
                    section_text = ''

                if len(child_text) > chunk_size:
                    sub_sections_as_text.extend(self._split_section_to_text(child, chunk_size))
                else:
                    sub_sections_as_text.append(child.parent_text() + "\n" + child_text)
            else:
                # group together paragraghs, tables, etc., everything that is not a section
                section_text += ("\n" if section_text else '') + child_text

        if section_text:
            sub_sections_as_text.append(section.parent_text() + "\n" + section.title + "\n" + section_text)
                
        return sub_sections_as_text


    def _split_document_to_text(self, doc, chunk_size=CHUNK_SIZE, first_n_chunks=None):
        """ Splits a document into chunks of text, where chunks are ideally sections of the document. 
        If a section is too large, it is recursively split into smaller subsections.
        The split algorithm attempts to preserve the section-level structure of the document as much as possible, to maintain the local context of the information present in the document.
        """
        chunks = []
        main_sections = [section for section in doc.sections() if section.level == 0]
        [chunks.extend(self._split_section_to_text(section, chunk_size=chunk_size)) for section in main_sections]

        if first_n_chunks and first_n_chunks < len(chunks):
            chunks = chunks[:first_n_chunks+1]

        return chunks

    def index_pdf(self, path_or_url, content=None, chunk_size=CHUNK_SIZE, first_n_chunks=None, add_summary=False,
                    retrieve_top_k=2, similarity_threshold=0.8):
        """
        Indexes a PDF document from a file path, URL, or bytes content.
        Initializes a query engine that will respond to user queries by retrieving similar chunks of text from the document, and passing them on to a LLM as context.

        Parameters
        ----------
        path_or_url: str
            path or url to the pdf file
        content: bytes
            contents of the pdf file. If content is given, path_or_url is ignored. This is useful when you already have the pdf file contents in memory such as if you are using streamlit or flask.
        chunk_size: int
            size of the chunks to split the document into. Default is 2048.
        first_n_chunks: int
            number of chunks to index. Default is None for indexing all document. Can be smaller than the total number of chunks in the document for test purposes.
        add_summary: bool
            whether to add a summary to each document chunk as metadata. Default is False.
        retrieve_top_k: int
            number of similar documents to retrieve for each query. Default is 2.
        similarity_threshold: float
            keep only documents with similarity score above this threshold. Default is 0.8.
        """
        if content is not None:
            doc = self.pdf_reader.read_pdf(path_or_url=path_or_url, contents=content)
        else:
            doc = self.pdf_reader.read_pdf(path_or_url)


        # split the document into chunks of text
        self.doc_chunks = self._split_document_to_text(doc, chunk_size=chunk_size, first_n_chunks=first_n_chunks)

        nodes = [Document(text=chunk_text, extra_info={}) for chunk_text in self.doc_chunks]

        # add summary metadata to each chunk
        if add_summary:
            metadata_extractors = [SummaryExtractor(summaries=["self"])]
            pipeline = IngestionPipeline(transformations=metadata_extractors)
            nodes = pipeline.run(nodes=nodes, in_place=False, num_workers=2, show_progress=True)

        # create index and retriever
        self._index = VectorStoreIndex(nodes)
        
        self._retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=retrieve_top_k,
        )

        self._response_synthesizer = get_response_synthesizer()

        # assemble query engine
        self._query_engine = RetrieverQueryEngine(
            retriever=self._retriever,
            response_synthesizer=self._response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_threshold)],
        )

    def get_query_engine(self):
        return self._query_engine
    
    def get_retriever(self):
        return self._retriever
    
    def update_similarity_threshold(self, similarity_threshold):
        # recreate query engine with new similarity threshold
        self._query_engine = RetrieverQueryEngine(
            retriever=self._retriever,
            response_synthesizer=self._response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_threshold)])
        return self._query_engine
    
    def update_similarity_top_k(self, similarity_top_k):
        # recreate retriever with new similarity_top_k
        self._retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=similarity_top_k,
        )
        # recreate query engine with new retriever
        self._query_engine = RetrieverQueryEngine(
            retriever=self._retriever,
            response_synthesizer=self._response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor()])
        return self._query_engine
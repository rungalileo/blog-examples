from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy

class SpacySentenceTokenizer():
    def __init__(self, spacy_model="en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self._chunk_size = None
        self._chunk_overlap = None

    def create_documents(self, documents, metadatas=None):
        chunks = []
        for doc, metadata in zip(documents, metadatas):
            for sent in self.nlp(doc).sents:
                chunks.append(Document(page_content=sent.text, metadata=metadata))
        return chunks

def get_indexing_configuration(config):
    if config == 1:
        text_splitter = SpacySentenceTokenizer()
        text_splitter_identifier = "sst"
        emb_model_name, dimension, emb_model_identifier = "text-embedding-3-small", 1536, "openai-small"
        embeddings = OpenAIEmbeddings(model=emb_model_name, tiktoken_model_name="cl100k_base")
        index_name = f"beauty-{text_splitter_identifier}-{emb_model_identifier}"
        
    elif config == 2:
        text_splitter = SpacySentenceTokenizer()
        text_splitter_identifier = "sst"
        emb_model_name, dimension, emb_model_identifier = "text-embedding-3-large", 1536*2, "openai-large"
        embeddings = OpenAIEmbeddings(model=emb_model_name, tiktoken_model_name="cl100k_base")
        index_name = f"beauty-{text_splitter_identifier}-{emb_model_identifier}"
        
    elif config == 3:
        text_splitter = SpacySentenceTokenizer()
        text_splitter_identifier = "sst"
        emb_model_name, dimension, emb_model_identifier = "all-MiniLM-L6-v2", 384, "all-minilm-l6"
        embeddings = HuggingFaceEmbeddings(model_name=emb_model_name, encode_kwargs = {'normalize_embeddings': True, 'show_progress_bar': False})
        index_name = f"beauty-{text_splitter_identifier}-{emb_model_identifier}"
        
    elif config == 4:
        text_splitter = SpacySentenceTokenizer()
        text_splitter_identifier = "sst"
        emb_model_name, dimension, emb_model_identifier = "all-mpnet-base-v2", 768, "all-mpnet"
        embeddings = HuggingFaceEmbeddings(model_name=emb_model_name, encode_kwargs = {'normalize_embeddings': True, 'show_progress_bar': False})
        index_name = f"beauty-{text_splitter_identifier}-{emb_model_identifier}"
        
    elif config == 5:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        text_splitter_identifier = "rc"
        emb_model_name, dimension, emb_model_identifier = "text-embedding-3-small", 1536, "openai-small"
        embeddings = OpenAIEmbeddings(model=emb_model_name, tiktoken_model_name="cl100k_base")
        index_name = f"beauty-{text_splitter_identifier}-cs{text_splitter._chunk_size}-co{text_splitter._chunk_overlap}-{emb_model_identifier}"
        
    elif config == 6:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
        text_splitter_identifier = "rc"
        emb_model_name, dimension, emb_model_identifier = "text-embedding-3-small", 1536, "openai-small"
        embeddings = OpenAIEmbeddings(model=emb_model_name, tiktoken_model_name="cl100k_base")
        index_name = f"beauty-{text_splitter_identifier}-cs{text_splitter._chunk_size}-co{text_splitter._chunk_overlap}-{emb_model_identifier}"

        
    return text_splitter, embeddings, emb_model_name, dimension, index_name
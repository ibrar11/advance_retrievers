import os
import json
from typing import List, Optional
import asyncio
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# Core LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings,
    DocumentSummaryIndex,
    KeywordTableIndex
)
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    AutoMergingRetriever,
    RecursiveRetriever,
    QueryFusionRetriever
)
from llama_index.core.indices.document_summary import (
    DocumentSummaryIndexLLMRetriever,
    DocumentSummaryIndexEmbeddingRetriever,
)
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from huggingface_hub import login
from dotenv import load_dotenv

# Advanced retriever imports
from llama_index.retrievers.bm25 import BM25Retriever

# Sentence transformers
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM


SAMPLE_DOCUMENTS = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
    "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "Reinforcement learning is a type of machine learning where agents learn to make decisions through rewards and penalties.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds hidden patterns in data without labeled examples.",
    "Transfer learning leverages knowledge from pre-trained models to improve performance on new tasks.",
    "Generative AI can create new content including text, images, code, and more.",
    "Large language models are trained on vast amounts of text data to understand and generate human-like text."
]

# Consistent query examples used throughout the lab
DEMO_QUERIES = {
    "basic": "What is machine learning?",
    "technical": "neural networks deep learning", 
    "learning_types": "different types of learning",
    "advanced": "How do neural networks work in deep learning?",
    "applications": "What are the applications of AI?",
    "comprehensive": "What are the main approaches to machine learning?",
    "specific": "supervised learning techniques"
}

# Statistical libraries for fusion techniques
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available - some advanced fusion features will be limited")

print("✅ All imports successful!")

load_dotenv()

def hf_login():
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_HUB_TOKEN not set")
    login(token=token)

def create_hf_llm():
    model_id = 'ibm-granite/granite-3.3-8b-instruct'
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        dtype='auto'
    )
    try:
        llm = HuggingFaceLLM(
            model_name=model_id,
            tokenizer_name=model_id,
            model=model,
            max_new_tokens=500,
            completion_to_prompt=None,
            messages_to_prompt=None,
            generate_kwargs={
                'temprature':0.1,
                'top_k':50,
                'top_p':1,
                'do_sample':'sample'
            }
        )
        print(f"Created HuggingFace LLM model: {model_id}")
        return llm
    except Exception as e:
        print("Error creating hugging face llm: ",e)
        return
    
def hh_embeddings():
    embeddings = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    return embeddings
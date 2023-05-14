# Dependencies for data processing and general machine learning implementations
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union

# Dependencies for vector search
import faiss
from sentence_transformers import SentenceTransformer

# FAISS vector search
class FAISSVectorEngine:
    '''
    FAISS is Facebook AI Similarity Search, a library that supports efficient vector similarity search across large sets of corpus.
    Reference source: https://github.com/facebookresearch/faiss
    Reference tutorial: https://www.pinecone.io/learn/faiss-tutorial/
    '''
    def __init__(self, model: str) -> None:
        if model is None:
            model = 'distilbert-base-nli-stsb-mean-tokens'
        else:
            if model != 'distilbert-base-nli-stsb-mean-tokens':
                raise ValueError('The specified model is currently not supported for use in vector search embedding')
        self.model_hf = SentenceTransformer(model) # mmodel to embed text into vectors using HuggingFace sentence transformer
        self.corpus = None
        self.stored_vectors = None
    
    # Store the corpus as vectors in memory
    def store(self, corpus: list[str], mode_args: dict[str, Union[str, int, float]]={'mode': 'base'}) -> None:
        '''
        This creates the vector index stored for vector search. 
        By default, mode_args is a dict object specifying the parameters of the base (non-optimised) vector search.
        Optionally, mode_args specifying "mode": "boost" is the boost (optimmised) vector search using index partioning.
        '''
        self.corpus = corpus # store original corpus texts for retrieval during search
        vectors = self.model_hf.encode(self.corpus) # embed corpus text into vectors
        self.stored_vectors = faiss.IndexFlatL2(vectors.shape[1]) # set FAISS index based on embedding dimension
        self.stored_vectors.add(vectors) # store vectors for search
        
        # Optimised FAISS vector search via index partitioning
        if mode_args['mode'] == 'boost':
            print('Boost (optimised) FAISS vector search not yet implemented')
            # TODO add index partitioning for FAISS search optimisation
            pass
        
    # Perform vector search of query against stored vectors and retrieve top k results
    def search(self, query: str, k: int) -> list[str]:
        D, I = self.stored_vectors.search(self.model_hf.encode([query]), k) # embed input vector then perform vector search
        return list(np.array(self.corpus)[I][0])

# Entry vector engine class           
class VectorEngine:
    '''
    Main vector engine class
    '''
    def __init__(self, model: str=None, source: str='faiss') -> None:
        self.source = source
        self.model = model
        
        # Accepted vectors search from sources
        self.supported_vector_search = [
            'faiss', 
            'pinecone',
        ]
        
        if self.source not in self.supported_vector_search:
            raise ValueError('The specified model is currently not supported in this package. Supported vector search engines are: ' + ', '.join([f"{m}" for m in self.supported_vector_search]))
        
        # FAISS vector search call
        if self.source == 'faiss':
            self.instance = FAISSVectorEngine(self.model)
            
        # Pinecone vector search call
        if self.source == 'pinecone':
            # TODO add Pinecone vecter search integration implementation
            print('Pinecone vector search integration not yet implemented')
            pass
            
    # Direct to the attribute of the sub vector engine class (attribute not found in the main vector engine class)
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)

# Calculate cosine similarity of two matrices    
def cosine_similarity(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(t1.view(1, -1), t2.view(1, -1))

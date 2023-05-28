from __future__ import annotations

# Dependencies for data processing and general machine learning implementations
import torch
import torch.nn.functional as F
from typing import Union

# Dependencies for vector search
from panml.core.clustering.faiss import FAISSVectorEngine
from panml.constants import SUPPORTED_VECTOR_SEARCH_SOURCES, SUPPORTED_EMBEDDING_MODELS

# Entry vector engine class           
class VectorEngine:
    '''
    Main vector engine class
    '''
    def __init__(self, model: str='all-MiniLM-L6-v2', source: str='faiss', api_key: str=None) -> None:
        self.source = source
        self.model = model
        self.api_key = api_key
        self.model_emb_source = None
        self.supported_vector_search = SUPPORTED_VECTOR_SEARCH_SOURCES # supported vectors search sources
        self.supported_embedding_models = SUPPORTED_EMBEDDING_MODELS # supported embedding models
        
        # General exceptions handling on user input
        if self.source not in self.supported_vector_search:
            raise ValueError('The specified vector search implementation is currently not supported in this package. Supported vector search implementations are: ' + ', '.join([f"{m}" for m in self.supported_vector_search]))
        if self.model not in self.supported_embedding_models['huggingface'] and \
            self.model not in self.supported_embedding_models['openai']:
            raise ValueError('The specified embedding model is currently not supported in this package. Supported embedding models for vector search are: ' + '\n' + str(self.supported_embedding_models))
        
        self.model_emb_source = [model_source for model_source, embedding_models in self.supported_embedding_models.items() if self.model in embedding_models][0]
        
        # FAISS vector search call
        if self.source == 'faiss':
            self.instance = FAISSVectorEngine(self.model, self.model_emb_source, self.api_key)
            
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

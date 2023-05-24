from __future__ import annotations

# Dependencies for data processing and general machine learning implementations
import torch
import torch.nn.functional as F
from typing import Union

# Dependencies for vector search
from panml.core.clustering.faiss import FAISSVectorEngine

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

        # Accepted vectors search from sources
        self.supported_vector_search = [
            'faiss', 
            'pinecone',
        ]
        self.supported_embedding_models = {
            'huggingface': [
                'all-MiniLM-L6-v2',
                'all-mpnet-base-v2',
                'all-distilroberta-v1'
                'nq-distilbert-base-v1',
                'paraphrase-albert-small-v2',
                'paraphrase-MiniLM-L3-v2',
                'paraphrase-MiniLM-L6-v2',
                'multi-qa-MiniLM-L6-cos-v1',
                'multi-qa-distilbert-cos-v1',
                'msmarco-MiniLM-L6-cos-v5',
                'distiluse-base-multilingual-cased-v1',
                'distiluse-base-multilingual-cased-v2',
                'paraphrase-multilingual-MiniLM-L12-v2',
                'paraphrase-multilingual-mpnet-base-v2',
                'distilbert-base-nli-stsb-mean-tokens',
            ],
            'openai': [
                'text-embedding-ada-002', 
            ],
        }
        
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

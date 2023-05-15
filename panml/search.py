# Dependencies for data processing and general machine learning implementations
import numpy as np
import pandas as pd
import torch
from typing import Union

# Dependencies for vector search
import faiss
from sentence_transformers import SentenceTransformer
import openai

# FAISS vector search
class FAISSVectorEngine:
    '''
    FAISS is Facebook AI Similarity Search, a library that supports efficient vector similarity search across large sets of corpus.
    Reference source: https://github.com/facebookresearch/faiss
    Reference tutorial: https://www.pinecone.io/learn/faiss-tutorial/
    '''
    def __init__(self, model: str, model_emb_source: str, api_key: str) -> None:
        self.model = model
        self.model_emb_source = model_emb_source
        if self.model_emb_source == 'huggingface':
            self.model_emb = SentenceTransformer(self.model) # Embedding model using HuggingFace sentence transformer
        elif self.model_emb_source == 'openai':
            self.model_emb = None # Embedding model not required with third party API functions
            if api_key is None:
                raise ValueError('api key has not been specified for OpenAI model call')
            openai.api_key = api_key # Set API key
        else:
            raise ValueError('Embedding model is not recognised or currently supported')
        self.corpus = None
        self.stored_vectors = None
        
    def _get_openai_embedding(self, text: str, model: str) -> list[str]:
        text = text.replace('\n', ' ')
        return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
    
    def _get_embedding(self, corpus: list[str], model_emb_source: str) -> np.array:
        if model_emb_source == 'huggingface':
            return self.model_emb.encode(corpus)
        elif model_emb_source == 'openai':
            df_corpus = pd.DataFrame({'corpus': corpus})
            return np.array(df_corpus['corpus'].apply(lambda x: self._get_openai_embedding(x, model=self.model)).tolist())

    # Store the corpus as vectors
    def store(self, corpus: list[str], mode_args: dict[str, Union[str, int, float]]={'mode': 'base'}) -> None:
        '''
        This creates the vector index stored for vector search. 
        By default, mode_args is a dict object specifying the parameters of the base (non-optimised) vector search.
        Optionally, mode_args specifying "mode": "boost" is the boost (optimmised) vector search using index partioning.
        '''
        if not isinstance(corpus, list):
            raise ValueError('Input text corpus needs to be of type list')
        if len(corpus) < 1:
            raise ValueError('Input text corpus is empty')
        
        self.corpus = corpus # store original corpus texts for retrieval during search
        vectors = self._get_embedding(self.corpus, self.model_emb_source) # embed corpus text into vectors
        self.stored_vectors = faiss.IndexFlatL2(vectors.shape[1]) # set FAISS index based on embedding dimension
        self.stored_vectors.add(vectors) # store vectors for search
        
        # Optimised FAISS vector search via index partitioning
        if mode_args['mode'] == 'boost':
            print('Boost (optimised) FAISS vector search not yet implemented')
            # TODO add index partitioning for FAISS search optimisation
            pass
        
    # Perform vector search of query against stored vectors and retrieve top k results
    def search(self, query: str, k: int) -> list[str]:
        if not isinstance(query, str):
            raise ValueError('Input text query needs to be of type str')
        if not isinstance(k, int):
            raise ValueError('Input number of returned documents needs to be of type int')
        if k > len(self.corpus):
            k = len(self.corpus) # cap the max k to the maximum number of documents in the corpus store
        query_vector = self._get_embedding([query], self.model_emb_source) # embed input vector
        D, I = self.stored_vectors.search(query_vector, k) # perform vector search
        return list(np.array(self.corpus)[I][0])

# Entry vector engine class           
class VectorEngine:
    '''
    Main vector engine class
    '''
    def __init__(self, model: str='distilbert-base-nli-stsb-mean-tokens', source: str='faiss', api_key: str=None) -> None:
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

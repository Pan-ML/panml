# Dependencies for data processing and general machine learning implementations
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Union

# Dependencies for vector search
import faiss
from sentence_transformers import SentenceTransformer
import openai
import pickle

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
    def store(self, corpus: list[str], save_name: str='stored',
              mode_args: dict[str, Union[str, int, float]]={
                  'mode': 'base', 
                }) -> None:
        '''
        This creates the vector index stored for vector search. 

        Args:
        corpus: list of texts containing the document contents forming the corpus to be searched
        save_name: custom name affix of the vectors and corpus assigned to the saved files (file type extension should not to be included)
        mode_args: parameters related to FAISS search, affecting different degrees of optimization

        Returns:
        None. Vectors store is setup, and if specified, the vectors and corpus are saved to files
        '''   
        # Catch input exceptions
        if not isinstance(corpus, list):
            raise TypeError('Input corpus needs to be of type: list')
        if not isinstance(mode_args, dict):
            raise TypeError('Input model args needs to be of type: dict')
        if len(corpus) < 1:
            raise ValueError('Input text corpus is empty')
        if save_name is not None:
            if not isinstance(save_name, str):
                raise TypeError('Input save name needs to be of type: str')
        
        self.corpus = corpus # store original corpus texts for retrieval during search
        vectors = self._get_embedding(self.corpus, self.model_emb_source) # embed corpus text into vectors
        self.stored_vectors = faiss.IndexFlatL2(vectors.shape[1]) # set FAISS index based on embedding dimension
        self.stored_vectors.add(vectors) # store vectors for search

        # Optimised FAISS vector search via index partitioning
        if mode_args['mode'] == 'boost':
            print('Boost (optimised) FAISS vector search not yet implemented')
            # TODO add index partitioning for FAISS search optimisation
            pass

        # Saving the vector store
        if save_name:
            faiss.write_index(self.stored_vectors, f"{save_name}_vectors.faiss") # vectors file
            with open(f"{save_name}_corpus.pkl", "wb") as f: # corpus file
                pickle.dump(self.corpus, f)
        
    # Load vectors and corpus
    def load(self, vectors_dir: str, corpus_dir: str) -> None:
        '''
        This loads the stored vectors (FAISS index) and the corresonding corpus (list object) for use in docs search
        
        Args: 
        vectors_dir: relative path of the stored vectors FAISS index file
        corpus_dir: relative path of the stored corpus pickle file

        Returns:
        None. Vectors and corpus are loaded into object attributes
        '''
        # Catch input exceptions
        if not isinstance(vectors_dir, str):
            raise TypeError('Input vectors dir needs to be of type: str')
        else:
            if not vectors_dir.endswith('faiss'):
                raise ValueError('Input vectors directory needs to end with .faiss')
        if not isinstance(corpus_dir, str):
            raise TypeError('Input corpus dir needs to be of type: str')
        else:
            if not corpus_dir.lower().endswith('pkl'):
                raise ValueError('Input corpus directory needs to end with .pkl')
        
        # Load the vectors and corpus
        self.stored_vectors = faiss.read_index(f"{vectors_dir}") # load vectors
        with open(f"{corpus_dir}", "rb") as f: # load corpus
            self.corpus = pickle.load(f)

    # Perform vector search of query against stored vectors and retrieve top k results
    def search(self, query: str, k: int) -> list[str]:
        '''
        Runs vector search of input query against the stored vectors.
        
        Args:
        query: text of the search query
        k: top number of simiilar documents w.r.t the search query

        Returns: 
        list of the top k documents
        '''
        # Catch input exceptions
        if not isinstance(query, str):
            raise TypeError('Input query needs to be of type: string')
        if not isinstance(k, int):
            raise TypeError('Input number of returned documents needs to be of type int')
            
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

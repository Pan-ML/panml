from __future__ import annotations

# Dependencies for data processing and general machine learning implementations
import numpy as np
import pandas as pd
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
    def store(self, corpus: list[str], model_name: str='stored', 
              mode: str='flat', nlist: int=None, m: int=None, bits: int=None) -> None:
        '''
        This creates the vector index stored for vector search. 

        Args:
        corpus: list of texts containing the document contents forming the corpus to be searched
        model_name: custom model name affix of the vectors and corpus assigned to the saved files (file type extension should not to be included)
        mode: parameter of FAISS vector search (flat, ivfflat, ivfpq) can improve speed, at the potentially the trade-off with accuracy
        nlist: parameter of mode: 'flat', specifies the number of partitions in the vectors space
        m: parameter of the mode: 'ivfpq', specifies the number of centroid IDs in compressed vectors
        bits: parameter of mode: 'ivfpq', specifies the number of bits in each centroid

        Returns:
        None. Vectors store is setup, and if specified, the vectors and corpus are saved to files
        '''   
        # Catch input exceptions
        if not isinstance(corpus, list):
            raise TypeError('Input corpus needs to be of type: list')
        if len(corpus) < 1:
            raise ValueError('Input text corpus is empty')
        if model_name is not None:
            if not isinstance(model_name, str):
                raise TypeError('Input save name needs to be of type: str')
        if not isinstance(mode, str):
            raise TypeError('Input mode needs to be of type: str')
        if mode not in ['flat', 'ivfflat', 'ivfpq']:
            raise ValueError('Supported mode options are: flat, ivfflat, ivfpq')
        if nlist is not None:
            if not isinstance(nlist, int):
                raise TypeError('Input nlist needs to be of type: int')
            if nlist < 1:
                raise ValueError('Input nlist needs to be greater or equal to 1')
        if m is not None:
            if not isinstance(m, int):
                raise TypeError('Input m needs to be of type: int')
            if m < 1:
                raise ValueError('Input m needs to be greater or equal to 1')
        if bits is not None:
            if not isinstance(bits, int):
                raise TypeError('Input bits needs to be of type: int')
            if bits < 1:
                raise ValueError('Input bits needs to be greater or equal to 1')
        if mode == 'ivfpq' and (m is None or bits is None):
            raise ValueError('Input m and bits are required for IVFPQ vector search')
        
        self.corpus = corpus # store original corpus texts for retrieval during search
        emb = self._get_embedding(self.corpus, self.model_emb_source) # embed corpus text into vectors
        
        if mode == 'flat': # brute force search with L2 measure (euclidean distance)
            self.vectors = faiss.IndexFlatL2(emb.shape[1])
            self.vectors.add(emb) # store vectors for search
            
        elif mode == 'ivfflat': # partitioned optimized search with L2 measure
            if nlist is None:
                nlist = int((0.5*emb.shape[1])**(1/2)) # default starting nlist value
            quantizer = faiss.IndexFlatL2(emb.shape[1])
            self.vectors = faiss.IndexIVFFlat(quantizer, emb.shape[1], nlist)
            self.vectors.train(emb)
            self.vectors.add(emb)
            self.vectors.make_direct_map() # map vectors for vectors retreival
            
        elif mode == 'ivfpq': # partitioned optimized search with L2 measure using compressed vectors 
            if m is not None or bits is not None:
                if nlist is None:
                    nlist = int((0.5*emb.shape[1])**(1/2))
                quantizer = faiss.IndexFlatL2(emb.shape[1])
                self.vectors = faiss.IndexIVFPQ(quantizer, emb.shape[1], nlist, m, bits)
                self.vectors.train(emb)
                self.vectors.add(emb)
                self.vectors.make_direct_map()
            else:
                raise ValueError('IVFPQ vector search not set due to missing m and bits inputs')

        # Saving the vector store
        if model_name:
            faiss.write_index(self.vectors, f"{model_name}_vectors.faiss") # vectors file
            with open(f"{model_name}_corpus.pkl", "wb") as f: # corpus file
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
        else:
            if len(query) < 1:
                raise ValueError('Input query cannot be empty')
        if not isinstance(k, int):
            raise TypeError('Input number of returned documents needs to be of type int')
        else:
            if k < 1:
                raise ValueError('Input k cannot be less than 1')
            
        if k > len(self.corpus):
            k = len(self.corpus) # cap the max k to the maximum number of documents in the corpus store
        query_vector = self._get_embedding([query], self.model_emb_source) # embed input vector
        D, I = self.vectors.search(query_vector, k) # perform vector search
        return list(np.array(self.corpus)[I][0])
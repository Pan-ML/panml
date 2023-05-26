from __future__ import annotations

# Dependencies for data processing and general machine learning implementations
import torch
from typing import Union

# Dependencies for specialised language model implementations
from transformers import AutoTokenizer
from panml.core.llm.huggingface import HuggingFaceModelPack
from panml.core.llm.openai import OpenAIModelPack
from panml.constants import SUPPORTED_LLMS, SUPPORTED_LLM_SOURCES

# Entry model pack class           
class ModelPack:
    '''
    Main model pack class
    '''
    def __init__(self, model: str, tokenizer: AutoTokenizer=None, input_block_size: int=20, padding_length: int=100, 
                 tokenizer_batch: bool=False, source: str='huggingface', api_key: str=None, model_args={}) -> None:
        self.padding_length = padding_length
        self.tokenizer = tokenizer
        self.model = model
        self.input_block_size = input_block_size
        self.tokenizer_batch = tokenizer_batch
        self.source = source
        self.api_key = api_key
        self.model_args = model_args
        self.supported_models = SUPPORTED_LLMS # supported models
        self.supported_sources = SUPPORTED_LLM_SOURCES # supported source descriptions
        
        # General exceptions handling on user input
        if self.source not in self.supported_sources:
            raise ValueError('The specified source is not recognized. Supported sources are: ' + ' '.join([f"{s}" for s in self.supported_sources]))

        # HuggingFace Hub model call
        if self.source == 'huggingface':
            if self.model not in self.supported_models['huggingface']:
                raise ValueError('The specified model is currently not supported in this package. Supported HuggingFace Hub models are: ' + ' '.join([f"{m}" for m in self.supported_models['huggingface']]))
            self.instance = HuggingFaceModelPack(self.model, self.input_block_size, self.padding_length, self.tokenizer_batch, self.source, self.model_args)

        # Locally trained model call of HuggingFace Hub model
        elif self.source == 'local':
            self.instance = HuggingFaceModelPack(self.model, self.input_block_size, self.padding_length, self.tokenizer_batch, self.source, self.model_args)

        # OpenAI model call
        elif self.source == 'openai':
            if self.model not in self.supported_models['openai']:
                raise ValueError('The specified model currently is not supported in this pacckage. Supported OpenAI models are: ' + ' '.join([f"{m}" for m in self.supported_models['openai']]))
            if self.api_key is None:
                raise ValueError('api key has not been specified for OpenAI model call')
            self.instance = OpenAIModelPack(model=self.model, api_key=self.api_key)

    # Direct to the attribute ofthe sub model pack class (attribute not found in the main model pack class)
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)
    
# Set device type
def set_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'

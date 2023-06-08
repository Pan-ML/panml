from __future__ import annotations

# Dependencies for specialised language model implementations
from panml.llm.huggingface import HuggingFaceModelPack
from panml.llm.openai import OpenAIModelPack
from panml.constants import SUPPORTED_LLMS, SUPPORTED_LLM_SOURCES
from panml.environments import ImportEnvironment

# Entry model pack class           
class ModelPack:
    '''
    Main model pack class
    '''
    def __init__(self, model: str, source: str='huggingface', model_args: dict={}, api_key: str=None) -> None:
        self.model = model
        self.source = source
        self.api_key = api_key
        self.model_args = model_args
        self.supported_models = SUPPORTED_LLMS # supported models
        self.supported_sources = SUPPORTED_LLM_SOURCES # supported source descriptions
        
        # General exceptions handling on user input
        if self.source not in self.supported_sources:
            raise ValueError('The specified source is not recognized. Supported sources are: ' + ' '.join([f"{s}" for s in self.supported_sources]))

        # HuggingFace Hub model processing
        if self.source == 'huggingface':
            ImportEnvironment(['transformers', 'datasets', 'peft', 'pandas', 'torch']).validate() # Validate required packages
            if self.model not in self.supported_models['huggingface']:
                raise ValueError('The specified model is currently not supported in this package. Supported HuggingFace Hub models are: ' + ' '.join([f"{m}" for m in self.supported_models['huggingface']]))
            self.instance = HuggingFaceModelPack(self.model, self.source, self.model_args)

        # Locally trained HuggingFace Hub model processing
        elif self.source == 'local':
            ImportEnvironment(['transformers', 'datasets', 'peft', 'pandas', 'torch']).validate()
            self.instance = HuggingFaceModelPack(self.model, self.source, self.model_args)

        # OpenAI model processing
        elif self.source == 'openai':
            ImportEnvironment(['openai', 'pandas', 'torch']).validate()
            if self.model not in self.supported_models['openai']:
                raise ValueError('The specified model currently is not supported in this package. Supported OpenAI models are: ' + ' '.join([f"{m}" for m in self.supported_models['openai']]))
            if self.api_key is None:
                raise ValueError('api key has not been specified for OpenAI model call')
            self.instance = OpenAIModelPack(model=self.model, api_key=self.api_key)

    # Direct to the attribute ofthe sub model pack class (attribute not found in the main model pack class)
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)
    

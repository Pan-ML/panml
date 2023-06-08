from __future__ import annotations
import pandas as pd
import openai
import torch
from typing import Union, Callable
from panml.constants import SUPPORTED_OAI_CODE_MODELS, SUPPORTED_OAI_COMPLETION_MODELS, SUPPORTED_OAI_CHAT_MODELS, SUPPORTED_EMBEDDING_MODELS

# OpenAI model class
class OpenAIModelPack:
    '''
    OpenAI model pack class
    '''
    def __init__(self, model: str, api_key: str) -> None:
        self.model_name = model # model name
        self.model_embedding = SUPPORTED_EMBEDDING_MODELS['openai'] # model embedding name
        self.supported_code_models = SUPPORTED_OAI_CODE_MODELS # supported code models
        self.completion_models = SUPPORTED_OAI_COMPLETION_MODELS # supported completion models
        self.chat_completion_models = SUPPORTED_OAI_CHAT_MODELS # supported chat models
        openai.api_key = api_key # set api key
        self.prediction_history = [] # model inference history in prompt loop
        
    def _init_prompt(self) -> list[dict[str, Union[str, Callable]]]:
        return [{'prepend': '', 'append': '', 'transform': None}]
    
    # Generate text of single model call
    def _predict(self, text: str, temperature: float, max_tokens: int, top_p: float, n: int, 
                 frequency_penalty: float, presence_penalty: float, display_probability: bool, logprobs: int, 
                 chat_role: str, stream: bool) -> dict[str, str]:
        '''
        Base function for text generation using OpenAI models

        Args:
        See predict function args
        
        Returns: 
        dict containing: 
        text: generated text
        probability: token probabilities if available
        perplexity: perplexity score if available
        Note: probability and peplexity scores are calculated only for some OpenAI models in this package
        '''
        output_context = {
            'text': None
        }
        
        if self.model_name in self.completion_models:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=text,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=logprobs,
                stream=stream,
            )
            
            # Return response stream
            if stream:
                return response
            
            output_context['text'] = response['choices'][0]['text']
            
        elif self.model_name in self.chat_completion_models:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages = [
                    {'role': chat_role, 'content': text}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=stream
            )

            # Return response stream
            if stream:
                return response
            
            output_context['text'] = response['choices'][0]['message']['content']
        else:
            raise ValueError(f'{self.model_name} is not currently supported in this package')

        # Get probability of output tokens
        if display_probability and self.model_name in self.completion_models:
            tokens = response["choices"][0]['logprobs']['tokens']
            token_logprobs = response["choices"][0]['logprobs']['token_logprobs']
            output_context['probability'] = [{'token': token, 'probability': torch.exp(torch.Tensor([logprob])).item()} for token, logprob in zip(tokens, token_logprobs)]

            # Calculate perplexity of output
            output_context['perplexity'] = (1/torch.prod(torch.tensor([torch.exp(torch.Tensor([logprob])).item() for logprob in token_logprobs])).item())**(1/len(tokens))

        return output_context
    
    # Generate text in prompt loop
    def predict(self, text: Union[str, list[str], pd.Series], temperature: float=0, max_length: int=100, top_p: float=1, n: int=3, 
                frequency_penalty: float=0, presence_penalty: float=0, display_probability: bool=False, logprobs: int=1, 
                prompt_modifier: list[dict[str, str]]=None, keep_history: bool=False, 
                chat_role: str='user', stream: bool=False) -> Union[dict[str, str], list[str]]:
        '''
        Generates output by prompting a language model from OpenAI
        
        Args:
        text: text of the prompt, can be a string, list, or pandas.series
        temperature: temperature of the generated text (for further details please refer to this topic covered in language model text generation)
        max_length: max number of tokens to be generated from OpenAI API
        top_p: max number of tokens to be generated from OpenAI API
        n: parameter from OpenAI API (for further details please refer to this topic covered in language model text generation)
        frequency_penalty: parameter from OpenAI API (for further details please refer to this topic covered in language model text generation)
        presence_penalty: parameter from OpenAI API (for further details please refer to this topic covered in language model text generation)
        display_probability: show probability of the generated tokens
        logprobs: parameter from OpenAI API (for further details please refer to this topic covered in language model text generation)
        prompt_modifier: list of prompts to be added in the context of multi-stage prompt chaining
        keep_history: keep or discard the history of responses from the mulit-stage prompt chaining
        chat_role: parameter from OpenAI API, specifies the role of the LLM assistant. Currently this is available for models of gpt-3.5-turbo or above (for further details please refer to this topic covered in language model text generation)
        stream: parameter from OpenAI API, specifies for streaming mode for text generation

        Returns: 
        prediction: list, or list of dict containing generated text with probabilities and perplexity if specified and available
        '''
        input_context = None
        self.prediction_history = []

        # Catch input exceptions
        if not isinstance(text, str) and not isinstance(text, list) and not isinstance(text, pd.Series):
            raise TypeError('Input text needs to be of type: string, list or pandas.series')
        if isinstance(text, pd.Series): # convert to list from pandas series if available
            input_context = text.tolist()
            if len(input_context) == 0:
                raise ValueError('Input text list cannot be empty')
        if isinstance(text, str): # wrap input text into list if available
            if len(text) == 0:
                raise ValueError('Input text cannot be empty')
            input_context = [text]
        if isinstance(text, list):
            if len(text) == 0:
                raise ValueError('Input text list cannot be empty')
            input_context = text
        if prompt_modifier is None:
            prompt_modifier = self._init_prompt()
        else:
            if not isinstance(prompt_modifier, list):
                raise TypeError('Input prompt modifier needs to be of type: list')
        if not isinstance(chat_role, str):
            raise TypeError('Input chat role needs to be of type: string')
        if not isinstance(stream, bool):
            raise TypeError('Input stream needs to be of type: boolean')
        else:
            if stream and not isinstance(text, str):
                raise ValueError('Streaming is not available for inputs (e.g. list or pandas dataframe column) that are outside of the string input use-case')
            
        # Run prediction on text samples
        prediction = []
        for context in input_context:
            # Create loop for text prediction
            history = []
            for count, mod in enumerate(prompt_modifier):
                # Set prepend or append to empty str if there is no input for these
                if 'prepend' not in mod: 
                    mod['prepend'] = ''
                if 'append' not in mod:
                    mod['append'] = ''
                if 'transform' not in mod:
                    mod['transform'] = None

                # Set the query text to previous output to carry on the prompt loop
                if count > 0:
                    context = output_context['text']
                if mod['transform'] is not None and callable(mod['transform']):
                    context = f"{mod['prepend']}\n{mod['transform'](context)}\n{mod['append']}"
                else:
                    context = f"{mod['prepend']}\n{context}\n{mod['append']}"

                if stream and count == len(prompt_modifier) - 1 and isinstance(text, str):
                    # Call model for text generation in streaming mode
                    output_context = self._predict(context, temperature=temperature, max_tokens=max_length, top_p=top_p,
                                                n=n, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                                                display_probability=display_probability, logprobs=logprobs, chat_role=chat_role, 
                                                stream=True)
                    return output_context
                else:
                    if stream:
                        print('Streaming is not applied. Streaming is only available for single prompt inputs.')
                    # Call model for text generation
                    output_context = self._predict(context, temperature=temperature, max_tokens=max_length, top_p=top_p,
                                                n=n, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                                                display_probability=display_probability, logprobs=logprobs, chat_role=chat_role, 
                                                stream=False)

                    # Terminate loop for next prompt when context contains no meaningful words (less than 2)
                    output_context['text'] = output_context['text'].replace('\n', ' ').replace('\xa0', '').replace('  ', ' ')
                    output_context['text'] = output_context['text'].replace(',', ', ')
                    output_context['text'] = output_context['text'].replace('.', '. ')
                    output_context['text'] = output_context['text'].replace('  ', ' ')
                    if len(output_context['text'].replace(' ', '')) < 2:
                        break

                    history.append(output_context)
            
                    try:
                        if keep_history:
                            prediction.append(history[-1]) # saves last prediction output
                            self.prediction_history.append(history) # saves all historical prediction output
                        else:
                            prediction.append(history[-1])
                    except:
                        prediction.append({'text': None}) # if there is invalid response from the language model, return None

        # Gather output
        if isinstance(text, str):
            return prediction[0] # return string text as result
        else:
            if display_probability:
                return prediction # return list of output_context dict as result
            else:
                return [pred.get('text', None) for pred in prediction] # return list as result
    
    # Generate and execute code using (LM powered function)
    def predict_code(self, text: str, x: Union[int, float, str, pd.DataFrame], variable_names: dict[str, str]={'input': 'x', 'output': 'y'}, 
                     language: str='python', max_tokens: int=500) -> str:
        '''
        Generates code output by prompting a language model from OpenAI with a constrained command that is specific for code generation.
        
        Args:
        text: text of the prompt
        x: input variable, can be a value or pandas dataframe
        variable_names: input variable name, and output variable name of the generated code
        langauge: programming language of the code to be generated
        max_tokens: max number of tokens to be generated from OpenAI API
   
        Returns: 
        text of generated code
        '''
        # Catch input exceptions
        if self.model_name not in self.supported_code_models:
            raise ValueError(f"The specified model is '{self.model_name}'. Only {', '.join(self.supported_code_models)} are supported in this package for code generation")
        if not isinstance(text, str):
            raise TypeError('Input text needs to be of type: string')
        if not isinstance(x, int) and not isinstance(x, float) and \
            not isinstance(x, str) and not isinstance(x, pd.DataFrame):
            raise TypeError('Input x needs to be of type: int, float, str or pandas dataframe')
        if not isinstance(variable_names, dict):
            raise TypeError('Input variable names needs to be of type: dict')
            
        if 'input' not in variable_names:
            variable_names['input'] = 'x'
        if 'output' not in variable_names:
            variable_names['output'] = 'y'
        
        # Prompt logic for code generation
        input_dataframe_col_names = ''
        if isinstance(x, pd.DataFrame):
            input_dataframe_col_names = f" dataframe has columns: {', '.join(x.columns.tolist())}"
            input_arg_context = f"{variable_names['output']} = None" # Set input and output variables
        else:
            input_arg_context = f"{variable_names['input']}, {variable_names['output']} = {x}, None" # Set output variables (if input is dataframe)

        prompt_prepend = f"Write {language} code to only produce function and input variable {variable_names['input']}" # Set instruction context
        prompt_append = f"of variable {variable_names['input']} and return in variable. {input_dataframe_col_names} \
                         Lastly, call function beginning with '{variable_names['output']} = .., and input of {variable_names['input']}'." # Set prompt details
        prompt_append = f"{prompt_append} do not show print, do not show enter input." # Set instruction to prevent code showing print or input functions
        prompt_append = f"{prompt_append} format the code with relevant indentation." # Set instruction to maintain code layout
        output_context = self.predict(f'{prompt_prepend} {text} {prompt_append}', max_tokens=max_tokens) # Get predicted code snippet
        code_context = f"{input_arg_context}\n{output_context['text']}" # Create code context

        return code_context
        
    # Embed text
    def embedding(self, text: str, model: str=None) -> list[float]:
        '''
        Gets the embedding of the text using OpenAI language models

        Args:
        text: text of the input
        model: name of the OpenAI language model (if not already provided when the class is instantiated)

        Returns:
        list containing the embedding array
        '''
        text = text.replace("\n", " ")
        if model is None:
            model = self.model_embedding          
        emb = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
        
        return emb
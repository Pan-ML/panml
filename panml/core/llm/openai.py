from __future__ import annotations
import pandas as pd
import torch
from typing import Union
import openai

# OpenAI model class
class OpenAIModelPack:
    '''
    OpenAI model pack class
    '''
    def __init__(self, model: str, api_key: str) -> None:
        self.model_name = model
        self.model_embedding = 'text-embedding-ada-002'
        openai.api_key = api_key
    
    # Generate text of single model call
    @staticmethod
    def _predict(model_name: str, text: str, temperature: float, max_tokens: int, top_p: float, n: int, 
                 frequency_penalty: float, presence_penalty: float, display_probability: bool, logprobs: int, 
                 chat_role: str) -> dict[str, str]:
        '''
        Base function for text generation using OpenAI models
        '''
        output_context = {
            'text': None,
            'probability': None,
            'perplexity': None,
        }
        completion_models = [
            'text-davinci-002', 
            'text-davinci-003',
        ]
        chat_completion_models = [
            'gpt-3.5-turbo',
            'gpt-3.5-turbo-0301',
        ]
        if model_name in completion_models:
            response = openai.Completion.create(
                model=model_name,
                prompt=text,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                logprobs=logprobs
            )
            output_context['text'] = response['choices'][0]['text']
            
        elif model_name in chat_completion_models:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages = [
                    {'role': chat_role, 'content': text}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            output_context['text'] = response['choices'][0]['message']['content']
        else:
            raise ValueError(f'{model_name} is not currently supported in this package')

        # Get probability of output tokens
        if display_probability and model_name in completion_models:
            tokens = response["choices"][0]['logprobs']['tokens']
            token_logprobs = response["choices"][0]['logprobs']['token_logprobs']
            output_context['probability'] = [{'token': token, 'probability': torch.exp(torch.Tensor([logprob])).item()} for token, logprob in zip(tokens, token_logprobs)]

            # Calculate perplexity of output
            output_context['perplexity'] = (1/torch.prod(torch.tensor([torch.exp(torch.Tensor([logprob])).item() for logprob in token_logprobs])).item())**(1/len(tokens))

        return output_context
    
    # Generate text in prompt loop
    def predict(self, text: str, temperature: float=0, max_tokens: int=100, top_p: float=1, n: int=3, 
                frequency_penalty: float=0, presence_penalty: float=0, display_probability: bool=False, logprobs: int=1, 
                prompt_modifier: list[dict[str, str]]=[{'prepend': '', 'append': ''}], keep_last: bool=True, 
                chat_role: str='user') -> dict[str, str]:
        '''
        Generates output by prompting a language model from OpenAI
        
        Args:
        text: text of the prompt
        temperature: temperature of the generated text (for further details please refer to this topic covered in language model text generation)
        max_tokens: max number of tokens to be generated from OpenAI API
        top_p: max number of tokens to be generated from OpenAI API
        n: parameter from OpenAI API (for further details please refer to this topic covered in language model text generation)
        frequency_penalty: parameter from OpenAI API (for further details please refer to this topic covered in language model text generation)
        presence_penalty: parameter from OpenAI API (for further details please refer to this topic covered in language model text generation)
        display_probability: show probability of the generated tokens
        logprobs: parameter from OpenAI API (for further details please refer to this topic covered in language model text generation)
        prompt_modifier: list of prompts to be added in the context of multi-stage prompt chaining
        keep_last: keep or discard the history of responses from the mulit-stage prompt chaining
        chat_role: parameter from OpenAI API, specifies the role of the LLM assistant. Currently this is available for models of gpt-3.5-turbo or above (for further details please refer to this topic covered in language model text generation)

        Returns: 
        dict containing: 
        text: generated text
        probability: token probabilities if available
        perplexity: perplexity score if available
        Note: probability and peplexity scores are calculated only for some OpenAI models in this package
        '''
        # Catch input exceptions
        if not isinstance(text, str):
            raise TypeError('Input text needs to be of type: string')
        if not isinstance(prompt_modifier, list):
            raise TypeError('Input prompt modifier needs to be of type: list')
        if not isinstance(chat_role, str):
            raise TypeError('Input chat role needs to be of type: string')
            
        # Create loop for text prediction
        response_words = 0
        history = []
        for count, mod in enumerate(prompt_modifier):
            # Set prepend or append to empty str if there is no input for these
            if 'prepend' not in mod: 
                mod['prepend'] = ''
            if 'append' not in mod:
                mod['append'] = ''

            # Set the query text to previous output to carry on the prompt loop
            if count > 0:
                text = output_context['text']
            text = f"{mod['prepend']} \n {text} \n {mod['append']}"

            # Call model for text generation
            output_context = self._predict(self.model_name, text, temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                                           n=n, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                                           display_probability=display_probability, logprobs=logprobs, chat_role=chat_role)

            # Terminate loop for next prompt when context contains no meaningful words (less than 2)
            response_words = output_context['text'].replace('\n', '').replace(' ', '')
            if len(response_words) < 2:
                break

            history.append(output_context)
        
        try:
            if keep_last:
                return history[-1] # returns last prediction output
            else:
                return history # returns all historical prediction output
        except:
            return {'text': None} # if there is invalid response from the language model, return None
        
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
        if self.model_name != 'text-davinci-002' and self.model_name != 'text-davinci-003':
            raise ValueError(f"The specified model is '{self.model_name}'. Only 'text-davinci-002' and 'text-davinci-002' are supported in this package for code generation")
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
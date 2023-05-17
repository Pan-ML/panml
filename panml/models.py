# Dependencies for data processing and general machine learning implementations
import pandas as pd
import torch
from typing import Union

# Dependencies for specialised language model implementations
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
import openai

# HuggingFace model class
class HuggingFaceModelPack:
    '''
    Generic HuggingFace Hub model pack class
    '''
    # Initialize class variables
    def __init__(self, model: str, input_block_size: int, padding_length: int, tokenizer_batch: bool, source: str) -> None:
        if source == 'huggingface':
            if 'flan' in model:
                self.model_hf = AutoModelForSeq2SeqLM.from_pretrained(model)
            else:
                self.model_hf = AutoModelForCausalLM.from_pretrained(model)
        elif source == 'local':
            if 'flan' in model:
                self.model_hf = AutoModelForSeq2SeqLM.from_pretrained(model, local_files_only=True)
            else:
                self.model_hf = AutoModelForCausalLM.from_pretrained(model, local_files_only=True)
        if self.model_hf.config.tokenizer_class:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf.config.tokenizer_class.lower().replace('tokenizer', ''), mirror='https://huggingface.co')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model, mirror='https://huggingface.co')
        self.padding_length = padding_length
        self.input_block_size = input_block_size
        self.tokenizer_batch = tokenizer_batch
        self.train_default_args = ['title', 'num_train_epochs', 'optimizer', 'mlm', 
                                   'per_device_train_batch_size', 'per_device_eval_batch_size',
                                   'warmup_steps', 'weight_decay', 'logging_steps', 
                                   'output_dir', 'logging_dir', 'save_model']
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.model_hf.config.eos_token_id
    
    # Embed text
    def embedding(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text, return_tensors='pt')
        
        # # Get embeddings
        # if 'flan' in self.model_hf.name_or_path: 
        #     emb = self.model_hf.shared.weight[token_ids[0]] # embeddings for FLAN
        # else: 
        #     emb = self.model_hf.transformer.wte.weight[token_ids[0]] # embeddings for GPT2
            
        # emb /= emb.norm(dim=1).unsqueeze(1) # normalise embedding weights
        # emb_pad = torch.zeros(self.padding_length, emb.shape[1]) # Set and apply padding to embeddings
        # emb_pad[:emb.shape[0], :] = emb

        model_input_params = {'input_ids': token_ids}
        if 'flan' in self.model_hf.name_or_path:
            model_input_params['decoder_input_ids'] = token_ids
        outputs = self.model_hf(**model_input_params, output_hidden_states=True) 
        return outputs.hidden_states[-1].mean(dim=1) # values in the last hidden layer averaged across all tokens
    
    # Generate text
    def predict(self, text: str, max_length: int=50, skip_special_tokens: bool=True, 
                display_probability: bool=False, num_return_sequences: int=1, temperature: float=0.8, 
                top_p: float=0.8, top_k: int=0, no_repeat_ngram_size: int=3) -> dict[str, str]:
        '''
        Generates output by prompting a language model from HuggingFace Hub
        Returns: dict of generated text (text), and token probabilities (probability) and perplexity scores (perplexity) if available
        '''
        # Catch input exceptions
        if not isinstance(text, str):
            raise TypeError('Input text needs to be of type: string')
            
        output_context = {
            'text': None,
            'probability': None,
            'perplexity': None,
        }
        
        input_ids = self.tokenizer.encode(text, return_tensors='pt')
        output = self.model_hf.generate(input_ids, 
                                        max_length=max_length,
                                        pad_token_id=self.model_hf.config.eos_token_id,
                                        num_return_sequences=num_return_sequences, 
                                        temperature=temperature,
                                        top_p=top_p,
                                        top_k=top_k,
                                        no_repeat_ngram_size=no_repeat_ngram_size,
                                        output_scores=display_probability, 
                                        return_dict_in_generate=display_probability, 
                                        renormalize_logits=True)
                                          
        # Get probability of output tokens
        if display_probability:
            output_context['text'] = self.tokenizer.decode(output['sequences'][0], skip_special_tokens=skip_special_tokens)
            output_context['probability'] = [
                {'token': self.tokenizer.decode(torch.argmax(torch.exp(s)).item()), 
                 'probability': torch.max(torch.exp(s)).item()} for s in output['scores']
            ]
            # Calculate perplexity of output
            output_context['perplexity'] = (1/(torch.prod(torch.Tensor([torch.max(torch.exp(s)).item() for s in output['scores']])).item()))**(1/len(output['scores']))
        else:
            output_context['text'] = self.tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
        output_context['text'] = output_context['text'].replace('\n', '')
        output_context['text'] = output_context['text'].strip()
        
        return output_context
    
    # Tokenize function
    def _tokenize_function(self, examples: Dataset) -> Dataset:
        return self.tokenizer(examples['text'])
    
    # Tokenize pandas dataframe feature
    def tokenize_text(self, x: list[str], batched: bool, num_proc: int) -> Dataset:  
        df_sample = pd.DataFrame({'text': x})
        hf_dataset = Dataset.from_pandas(df_sample)
        if batched:
            tokenized_dataset = hf_dataset.map(self._tokenize_function, batched=batched, num_proc=num_proc)
        else:
            tokenized_dataset = hf_dataset.map(self._tokenize_function)

        return tokenized_dataset
    
    # Model training
    def fit(self, x: list[str], y: list[str], train_args: dict[str, Union[str, int, float]]={}, 
            instruct: bool=False, num_proc=4) -> None:
        '''
        Fine tuning of a language model from HuggingFace Hub
        Returns: None. Trained model is saved in the .result/ folder with name "model_" prepended to the specified title
        '''
        # Catch input exceptions
        if not isinstance(x, list):
            raise TypeError('Input data array, x, needs to be of type: list')
        if not isinstance(y, list):
            raise TypeError('Input data array, y, needs to be of type: list')
        if not isinstance(train_args, dict):
            raise TypeError('Input train args needs to be of type: dict')
            
        # Convert to tokens format from pandas dataframes
        tokenized_data = self.tokenize_text(x, batched=self.tokenizer_batch, num_proc=num_proc)
        tokenized_target = self.tokenize_text(y, batched=self.tokenizer_batch, num_proc=num_proc)
        
        # Check for missing input arguments
        if set(list(train_args.keys())) != set(self.train_default_args):
            raise ValueError(f'Train args are not in the required format - missing: {", ".join(list(set(self.train_default_args) - set(list(train_args.keys()))))}')
        
        if instruct:
            print('Setting up training in sequence to sequence format...')
            tokenized_data = tokenized_data.add_column('labels', tokenized_target['input_ids']) # Create target sequence labels
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model) # Organise data for training
            
            # Setup training in sequence to sequence format
            training_args = TrainingArguments(
                optim=train_args['optimizer'], # model optimisation function
                num_train_epochs=train_args['num_train_epochs'], # total number of training epochs
                per_device_train_batch_size=train_args['per_device_train_batch_size'],  # batch size per device during training
                per_device_eval_batch_size=train_args['per_device_eval_batch_size'],   # batch size for evaluation
                warmup_steps=train_args['warmup_steps'], # number of warmup steps for learning rate scheduler
                weight_decay=train_args['weight_decay'], # strength of weight decay
                logging_steps=train_args['logging_steps'],
                output_dir='./results', # output directory
                logging_dir=train_args['logging_dir'], # log directory
            )

            trainer = Seq2SeqTrainer(
                model=self.model_hf,
                args=training_args,
                train_dataset=tokenized_data.remove_columns(['text']),
                eval_dataset=tokenized_data.remove_columns(['text']),
                data_collator=data_collator,
            )
    
        else:
            print('Setting up training in autoregressive format...')
            data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=train_args['mlm']) # Organise data for training
            
            # Setup training in autoregressive format
            training_args = TrainingArguments(
                optim=train_args['optimizer'], 
                num_train_epochs=train_args['num_train_epochs'], 
                per_device_train_batch_size=train_args['per_device_train_batch_size'],
                per_device_eval_batch_size=train_args['per_device_eval_batch_size'], 
                warmup_steps=train_args['warmup_steps'], 
                weight_decay=train_args['weight_decay'], 
                logging_steps=train_args['logging_steps'],
                output_dir=train_args['output_dir'],
                logging_dir=train_args['logging_dir'], 
            )
            
            trainer = Trainer(
                model=self.model_hf,
                args=training_args,
                train_dataset=tokenized_data.remove_columns(['text']),
                eval_dataset=tokenized_data.remove_columns(['text']),
                data_collator=data_collator,
            )

        trainer.train() # Execute training
        
        if train_args['save_model']:
            trainer.save_model(f'./results/model_{train_args["title"]}') # Save trained model
    
# OpenAI model class
class OpenAIModelPack:
    '''
    Generic OpenAI model pack class
    '''
    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.model_embedding = 'text-embedding-ada-002'
        openai.api_key = api_key
    
    # Generate text of single model call
    @staticmethod
    def _predict(model: str, text: str, temperature: float, max_tokens: int, top_p: float, n: int, 
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
        if model in completion_models:
            response = openai.Completion.create(
                model=model,
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
            
        elif model in chat_completion_models:
            response = openai.ChatCompletion.create(
                model=model,
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
            raise ValueError(f'{model} is not currently supported in this package')

        # Get probability of output tokens
        if display_probability and model in completion_models:
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
        Returns: dict of generated text (text), and token probabilities (probability) and perplexity scores (perplexity) if available
        Note: probability and peplexity scores are not calculated for gpt-3.5-turbo models
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
            output_context = self._predict(self.model, text, temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                                           n=n, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
                                           display_probability=display_probability, logprobs=logprobs, chat_role=chat_role)

            # Terminate loop for next prompt when context contains no meaningful words (less than 2)
            response_words = output_context['text'].replace('\n', '').replace(' ', '')
            if len(response_words) < 2:
                break

            history.append(output_context)
        
        if keep_last:
            return history[-1] # returns last prediction output
        else:
            return history # returns all historical prediction output
        
    # Generate and execute code using (LM powered function)
    def predict_code(self, text: str, x: Union[int, float, str, pd.DataFrame], variable_names: dict[str, str]={'input': 'x', 'output': 'y'}, 
                     language: str='python', max_tokens: int=500) -> str:
        '''
        Generates code output by prompting a language model from OpenAI with a constrained command that is specific for code generation.
        Input variables can be a value, or a pandas dataframe
        Variable names are passed into the prompt context to allow the generated code to contain the specified variable names.
        Returns: text of generated code
        '''
        # Catch input exceptions
        if self.model != 'text-davinci-002':
            raise ValueError(f"The specified model is '{self.model}'. Only 'text-davinci-002' is supported in this package for code generation")
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
        text = text.replace("\n", " ")
        if model is None:
            model = self.model_embedding          
        emb = openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
        
        return emb
        
# Entry model pack class           
class ModelPack:
    '''
    Main model pack class
    '''
    def __init__(self, model: str, tokenizer: AutoTokenizer=None, input_block_size: int=20, padding_length: int=100, 
                 tokenizer_batch: bool=False, source: str='huggingface', api_key: str=None) -> None:
        self.padding_length = padding_length
        self.tokenizer = tokenizer
        self.model = model
        self.input_block_size = input_block_size
        self.tokenizer_batch = tokenizer_batch
        self.source = source
        self.api_key = api_key
        
        # Accepted models from sources
        self.supported_models = {
            'huggingface': [
                'distilgpt2', 
                'gpt2',
                'gpt2-medium',
                'gpt2-xl',
                'google/flan-t5-base',
                'google/flan-t5-small',
                'google/flan-t5-large',
                'google/flan-t5-xl',
                'google/flan-t5-xxl',
                'cerebras/Cerebras-GPT-111M',
                'cerebras/Cerebras-GPT-256M',
                'cerebras/Cerebras-GPT-590M',
                'cerebras/Cerebras-GPT-1.3B',
                'cerebras/Cerebras-GPT-2.7B',
                'cerebras/Cerebras-GPT-6.7B',
                'cerebras/Cerebras-GPT-13B',
                'EleutherAI/gpt-neo-2.7B',
                'EleutherAI/gpt-j-6B',
                'StabilityAI/stablelm-base-alpha-3b',
                'StabilityAI/stablelm-base-alpha-7b',
                'StabilityAI/stablelm-tuned-alpha-3b',
                'StabilityAI/stablelm-tuned-alpha-7b',
                'h2oai/h2ogpt-oasst1-512-12b',
                'h2oai/h2ogpt-oasst1-512-20b',
                'Salesforce/codegen-350M-multi',
                'Salesforce/codegen-2B-multi',
            ],
            'openai': [
                'text-davinci-002', 
                'text-davinci-003',
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-0301',
            ],
        }

        # Accepted source descriptions
        self.supported_sources = [
            'huggingface', 
            'local', 
            'openai',
        ]
        
        # General exceptions handling on user input
        if self.source not in self.supported_sources:
            raise ValueError('The specified source is not recognized. Supported sources are: ' + ' '.join([f"{s}" for s in self.supported_sources]))

        # HuggingFace Hub model call
        if self.source == 'huggingface':
            if self.model not in self.supported_models['huggingface']:
                raise ValueError('The specified model is currently not supported in this package. Supported HuggingFace Hub models are: ' + ' '.join([f"{m}" for m in self.supported_models['huggingface']]))
            self.instance = HuggingFaceModelPack(self.model, self.input_block_size, self.padding_length, self.tokenizer_batch, self.source)

        # Locally trained model call of HuggingFace Hub model
        elif self.source == 'local':
            self.instance = HuggingFaceModelPack(self.model, self.input_block_size, self.padding_length, self.tokenizer_batch, self.source)

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

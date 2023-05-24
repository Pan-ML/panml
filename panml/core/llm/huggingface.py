from __future__ import annotations
import pandas as pd
import torch
from typing import Union
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM, AutoTokenizer
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from datasets import Dataset

# HuggingFace model class
class HuggingFaceModelPack:
    '''
    HuggingFace Hub model pack class
    '''
    # Initialize class variables
    def __init__(self, model: str, input_block_size: int, padding_length: int, tokenizer_batch: bool, source: str, model_args: dict) -> None:
        self.model_name = model
        self.padding_length = padding_length
        self.input_block_size = input_block_size
        self.tokenizer_batch = tokenizer_batch
        self.device = 'cpu'
        self.train_default_args = ['title', 'num_train_epochs', 'optimizer', 'mlm', 
                                   'per_device_train_batch_size', 'per_device_eval_batch_size',
                                   'warmup_steps', 'weight_decay', 'logging_steps', 
                                   'output_dir', 'logging_dir', 'save_model']
        if source == 'huggingface':
            if 'flan' in self.model_name:
                self.model_hf = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_args)
            elif 'bert' in self.model_name:
                self.model_hf = AutoModelForMaskedLM.from_pretrained(self.model_name, **model_args)
            else:
                self.model_hf = AutoModelForCausalLM.from_pretrained(self.model_name, **model_args)
        elif source == 'local':
            if 'flan' in self.model_name:
                self.model_hf = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_args, local_files_only=True)
            elif 'bert' in self.model_name:
                self.model_hf = AutoModelForMaskedLM.from_pretrained(self.model_name, **model_args, local_files_only=True)
            else:
                self.model_hf = AutoModelForCausalLM.from_pretrained(self.model_name, **model_args, local_files_only=True)
        if self.model_hf.config.tokenizer_class:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf.config.tokenizer_class.lower().replace('tokenizer', ''), mirror='https://huggingface.co')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, mirror='https://huggingface.co')

        # Set model on GPU if available and specified
        if 'gpu' in model_args:
            self.device = 'cuda' if torch.cuda.is_available() and model_args['gpu'] else 'cpu'
            print('Model processing is set on GPU')
        else:
            print('Model processing is set on CPU')
        self.model_hf.to(torch.device(self.device))

    # Embed text
    def embedding(self, text: str) -> torch.Tensor:
        '''
        Gets the embedding of the text using open source collections of models from HuggingFace Hub

        Args:
        text: text of the input

        Returns:
        torch tensor containing the embedding array
        '''
        token_ids = self.tokenizer.encode(text, return_tensors='pt').to(torch.device(self.device))
        
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
        hidden_states = 'decoder_hidden_states' if 'flan' in self.model_hf.name_or_path else 'hidden_states'
        return outputs[hidden_states][-1].mean(dim=1) # values in the last hidden layer averaged across all tokens
    
    # Generate text
    def predict(self, text: str, max_length: int=50, skip_special_tokens: bool=True, 
                display_probability: bool=False, num_return_sequences: int=1, temperature: float=0.8, 
                top_p: float=0.8, top_k: int=0, no_repeat_ngram_size: int=3) -> dict[str, str]:
        '''
        Generates output by prompting a language model from HuggingFace Hub

        Args:
        text: text of the prompt
        max_length: parameter from HuggingFace Hub, specifies the max length of tokens generated
        skip_special_tokens: parameter from HuggingFace Hub, specifies whether or not to remove special tokens in the decoding
        display_probability: show probability of the generated tokens
        num_return_sequences: parameter from HuggingFace Hub, specifies the number of independently computed returned sequences for each element in the batch
        temperature: parameter from HuggingFace Hub, specifies the value used to modulate the next token probabilities
        top_p: parameter from HuggingFace Hub, if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation
        no_repeat_ngram_size: parameter from HuggingFace Hub, ff set to int > 0, all ngrams of that size can only occur once

        Returns: 
        dict containing: 
        text: generated text
        probability: token probabilities if available
        perplexity: perplexity score if available
        '''
        # Catch input exceptions
        if not isinstance(text, str):
            raise TypeError('Input text needs to be of type: string')
            
        output_context = {
            'text': None,
            'probability': None,
            'perplexity': None,
        }
        
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(torch.device(self.device))
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
    def fit(self, x: Union[list[str], pd.Series], y: Union[list[str], pd.Series], train_args: dict[str, Union[str, int, float]]={}, 
            instruct: bool=False, num_proc: int=4) -> None:
        '''
        Fine tuning of a language model from HuggingFace Hub

        Args:
        x: list of example text for training/fine tuning the language model
        y: list of example text as targets for training/fine tuning the language model. 
        Note: For autogressive based training, this parameter is set to be same as x. For instruct based training, this parameter is part of the target examples for supervised fine tuning.
        train_args: parameters to be fed into HuggingFace Hub's training_args dict, as required for training
        num_proc: parameter from HuggingFace Hub, specifies max number of processes when generating cache. Already cached shards are loaded sequentially

        Returns: 
        None. Trained model is saved in the .result/ folder with name "model_" prepended to the specified title
        '''
        # Catch input exceptions
        if isinstance(x, pd.Series): # convert to list from pandas series if available
            x = x.tolist()
        if not isinstance(x, list):
            raise TypeError('Input data array, x, needs to be of type: list')
        if isinstance(y, pd.Series): # convert to list from pandas series if available
            y = y.tolist()
        if not isinstance(y, list):
            raise TypeError('Input data array, y, needs to be of type: list')
        if not isinstance(train_args, dict):
            raise TypeError('Input train args needs to be of type: dict')
        if not isinstance(num_proc, int):
            raise TypeError('Input num proc needs to be of type: int')
        else:
            if num_proc < 1:
                raise ValueError('Input num proc cannot be less than 1')
            
        # Convert to tokens format from pandas dataframes
        tokenized_data = self.tokenize_text(x, batched=self.tokenizer_batch, num_proc=num_proc)
        tokenized_target = self.tokenize_text(y, batched=self.tokenizer_batch, num_proc=num_proc)
        
        # Check for missing input arguments
        if set(list(train_args.keys())) != set(self.train_default_args):
            raise ValueError(f'Train args are not in the required format - missing: {", ".join(list(set(self.train_default_args) - set(list(train_args.keys()))))}')
        
        if instruct:
            print('Setting up training in sequence to sequence format...')
            tokenized_data = tokenized_data.add_column('labels', tokenized_target['input_ids']) # Create target sequence labels
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer) # Organise data for training
            
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
            if 'bert' in self.model_hf.name_or_path: # force training task to be mlm for encoders
                train_args['mlm'] = True
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

    # Save model
    def save(self, save_dir: str=None) -> None:
        '''
        Save the model on demand

        Args:
        save_dir: relative path of the file. If directory is not provided, the default directory is set to ".results/model_<model_name>"

        Returns:
        None. File is saved at the specified location
        '''
        if save_dir is None:
            save_dir = f'./results/model_{self.model_name}'
        self.model_hf.save_pretrained(save_dir)

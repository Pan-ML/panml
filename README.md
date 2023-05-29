## PanML: A high level generative AI/ML development and analysis library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![](https://dcbadge.vercel.app/api/server/cCsaqv9KFf?compact=true&style=flat)](https://discord.gg/cCsaqv9KFf)

## Goal
This package aims to make analysis and experimentation of generative AI/ML models broadly accessible, by providing a simple and consistent interface to foundation models, and abstract methods to support some of the common use-cases. This includes using smaller, less compute heavy Open Source language models to support the various NLP-based Data Science workflows in the industry. Additionally, we want to empower Data Science projects with the tools to easily productionise custom-built generative models.


We are passionate about AI technology and AI safety, and this supports our contribution towards a beneficial outcome in an AI-powered world. Please note this is a work in progress, so very much open for collaboration and contribution. 

### What this library covers
- [Inference and analysis of LLM](https://github.com/Pan-ML/panml/wiki/5.-Generative-model-analysis)
- [Fine tuning of LLM](https://github.com/Pan-ML/panml/wiki/3.-Fine-tuning-your-LLM)
- [Prompt chain engineering a LLM](https://github.com/Pan-ML/panml/wiki/2.-Prompt-chain-engineering)
- [Document question answering using LLM](https://github.com/Pan-ML/panml/wiki/7.-Retrieve-similar-documents-using-vector-search)
- [Variable integrated code generation using LLM](https://github.com/Pan-ML/panml/wiki/4.-Prompted-code-generation)

### Current supported foundation models
- [HuggingFace Hub](https://huggingface.co) - open source LLMs from Google, EleutherAI, Cerebras, StabilityAI, H2O, Salesforce, and others
- [OpenAI](https://openai.com) - text-davinci-002/003, GPT3/3.5

See model options in [library supported models](https://github.com/Pan-ML/panml/wiki/8.-Supported-models)

### Roadmap

Model development and analysis
- Support for additional open source and commercial language model
- Model evaluation methods (including adversarial evaluation)
- Prompt engineering support and analysis (adding further methods and adding support for more LLMs)
- Parameter-efficient fine tuning (e.g. LoRA, prompt tuning etc)
- Model intepretability methods
- Model emergent effect simulation and analysis (related to long term AI safety risks)

Model productionisation
- Automated refactoring of experimental code into source code
- Automated API wrapper generation
- Automated dockerization

### Support

You can support us by contributing to this project, as well as providing feedback and ideas in the [issues](https://github.com/Pan-ML/panml/issues) section. 

We would also appreciate if you can give panml a ⭐ on GitHub, and if it adds value to you, sharing this with others in your network on LinkedIn/Twitter/Medium etc who would also find this useful.


## Installation
```bash
pip install panml
```

## Usage
See [quick start guide](https://github.com/Pan-ML/panml/wiki/1.-Quick-start-guide) or detailed examples in the [PanML Wiki](https://github.com/Pan-ML/panml/wiki).

### Importing the module
```python
# Import panml
from panml.models import ModelPack

# Import other modules/packages as required
import numpy as np
import pandas as pd
...
```

### Using open source models from HuggingFace Hub
See model options in [library supported models](https://github.com/Pan-ML/panml/wiki/8.-Supported-models).
```python
lm = ModelPack(model='gpt2', source='huggingface')

output = lm.predict('hello world is')
print(output['text'])
```
```
'hello world is a place where people can live and work together, and where people can live and work together, and where people can live and work together'
```
Run inference on batch of inputs in dataframe
```python
df = pd.DataFrame({'input_prompts': [
    'The goal of life is',
    'The goal of work is',
    'The goal of leisure is',
]})

output = lm.predict(df['input_prompts'], max_length=20)
```
```
['The goal of life is to be a',
 'The goal of work is to make a living. ',
 'The goal of leisure is to get to the end of the']
```
When set to return token probability and perplexity scores
```python
output = lm.predict(df_test['prompts'], max_length=20, display_probability=True)
df_output = pd.DataFrame(output) # df_output contains columns: text, probability, perplexity
```

### Using open source models from OpenAI
```python
lm = ModelPack(model='text-davinci-003', source='openai', api_key=<your_openai_key>)

df = pd.DataFrame({'input_prompts': [
    'The goal of life is',
    'The goal of work is',
    'The goal of leisure is',
]})

output = lm.predict(df['input_prompts'])
```
```
[' to live a life of purpose, joy, and fulfillment. To find meaning and purpose in life, it is important to focus on what brings you joy and fulfillment, and to strive to make a positive impact on the world. It is also important to take care of yourself and your relationships, and to be mindful of the choices you make. ',
 ' The goal of this work is to develop a comprehensive understanding of a particular topic or issue, and to use that understanding to create solutions or strategies that can be implemented to address the issue. ',
 ' to provide an enjoyable and fulfilling experience that helps to reduce stress, improve physical and mental health, and promote social interaction. Leisure activities can include anything from physical activities such as sports and outdoor recreation, to creative activities such as art and music, to social activities such as attending events or visiting friends. ']
```

### Fine tune custom LLM
For detailed examples, see [fine tuning your LLM](https://github.com/Pan-ML/panml/wiki/3.-Fine-tuning-your-LLM). 
```python
# Load model
lm = ModelPack(model='google/flan-t5-base', source='huggingface', model_args={'gpu': True})

# Specify train args
train_args = {
    'title': 'my_tuned_flan_t5',
    'num_train_epochs' : 1,
    'mlm': False,
    'optimizer': 'adamw_torch',
    'per_device_train_batch_size': 10,
    'per_device_eval_batch_size': 10,
    'warmup_steps': 20,
    'weight_decay': 0.01,
    'logging_steps': 10,
    'output_dir': './results',
    'logging_dir': './logs',
    'save_model': True,
}

# Prepare data
x = df['input_text']
y = df['target_text']

# Train model
lm.fit(x, y, train_args, instruct=True)
```

### Prompt chain engineering
For detailed examples, see [prompt chain engineering](https://github.com/Pan-ML/panml/wiki/2.-Prompt-chain-engineering). <br>
Create model pack from OpenAI model description and API key.
```python
lm = ModelPack(model='text-davinci-002', source='openai', api_key=<your_openai_key>)

prompts = [
    {'prepend': 'you are a sports coach'},
    {'prepend': 'produce a daily exercise plan for one week'},
    {'prepend': 'summarise to the original question'},
]

output = lm.predict('What is the best way to live a healthy lifestyle?', prompt_modifier=prompts, max_tokens=600)
output['text']
```
```
'Assuming you are starting from a sedentary lifestyle, a good goal to aim for is 
30 minutes of moderate-intensity exercise most days of the week. 
This could include brisk walking, biking, swimming, or using a elliptical trainer. 
Start with whatever you feel comfortable with and gradually increase your time and intensity as you get more fit. 
Remember to warm up and cool down for 5-10 minutes before and after your workout. 
In addition to aerobic exercise, it is also important to include strength training in your routine. 
Strength-training not only helps to tone your body, but can also help to reduce your risk of injuries in the future. 
A simple way to start strength-training is to use your own body weight for resistance. 
Try doing push-ups, sit-ups, and squats. As you get stronger, you can add weight by using dumbbells or resistance bands. 
Aim for two to three days of strength-training per week. 
Finally, be sure to get enough sleep each night. Most adults need 7-8 hours of sleep per night. 
Getting enough sleep will help your body to recover from your workouts and will also help to reduce stress levels.'
```

### Prompted code generation
For detailed examples, see [prompted code generation](https://github.com/Pan-ML/panml/wiki/4.-Prompted-code-generation).
```python
code = lm.predict_code('calculate the fibonacci sequence using input', x=19, 
                       variable_names={'output': 'ans'}, language='python')
print(code)
exec(code) # execute code in Python
print(f'\nAnswer: {ans}')
```
```
x, ans = 19, None

def Fibonacci(x): 
    if x<0: 
        print("Incorrect input") 
    elif x==1: 
        return 0
    elif x==2: 
        return 1
    else: 
        return Fibonacci(x-1)+Fibonacci(x-2) 

ans = Fibonacci(x)

Answer: 2584
```

## Contributing

### Pre-requisites
- python 3.9x

### Setting up

```bash
git clone https://github.com/Pan-ML/panml.git

# create virtual environment
python3 -m venv . 

# activate virtual env - https://docs.python.org/3/library/venv.html#how-venvs-work
source bin/activate # if unix
Scripts/activate.bat # if windows cmd

# Install requirements
pip install -r requirements.txt
```

### Running tests
```
python3 -m unittest
```

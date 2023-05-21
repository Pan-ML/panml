## PanML: A high level generative AI/ML development library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![](https://dcbadge.vercel.app/api/server/QpquRMDq?compact=true&style=flat)](https://discord.gg/QpquRMDq)

## Goal
This package aims to make analysis and experimentation of generative AI/ML models more accessible, by providing a simple and consistent interface to foundation models for Data Scientists, Machine Learning Engineers and Software Developers. It's a work in progress, so very much open for collaboration and contribution. 
<br><br>
**Current supported generative AI/ML category** <br>
*Language models (fine tuning, prompt engineering, prompt tuning, model evaluation)*
<br><br>
**Current supported foundation models** <br>
*[HuggingFace Hub](https://huggingface.co) - open source LLMs from Google, EleutherAI, Cerebras, StabilityAI, H2O, Salesforce, and others, see [supported models](https://github.com/Pan-ML/panml/wiki/8.-Supported-models)* <br>
*[OpenAI](https://openai.com) - text-davinci-002/003, GPT3/3.5, see [supported models](https://github.com/Pan-ML/panml/wiki/8.-Supported-models) *
<br><br>
**Current supported evals** <br>
*Coming later...*
<br>

## Installation
```bash
git clone https://github.com/Pan-ML/panml.git
```

## Usage
See [quick start guide](https://github.com/Pan-ML/panml/wiki/1.-Quick-start-guide) or other examples in [Wiki](https://github.com/Pan-ML/panml/wiki).

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
```python
lm = ModelPack(model='gpt2', source='huggingface')

output = lm.predict('hello world is')
print(output['text'])
```
```
# Output
'hello world is a place where people can live and work together, and where people can live and work together, and where people can live and work together'
```

### Fine tune custom LLM
For more examples, see [fine tuning your LLM](https://github.com/Pan-ML/panml/wiki/3.-Fine-tuning-your-LLM).
```python
# Specify train args
train_args = {
    'title': 'my_tuned_gpt2',
    'num_train_epochs' : 5,
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
x = df['some_text']
y = x

# Train model
lm.fit(x, y, train_args, instruct=False)
```

### Prompt chain engineering
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
# Output
'\nAssuming you are starting from a sedentary lifestyle, a good goal to aim for is 
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

# Contributing

## Pre-requisites
- python 3.9x

## Setting up

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

## Running tests
```
python3 -m unittest
```

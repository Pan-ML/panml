## PanML: A simple generative AI/ML development toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![](https://4.vercel.app/github/language/:owner/:repo)]
[![](https://dcbadge.vercel.app/api/server/6adMQxSpJS?compact=true&style=flat)](https://discord.gg/QpquRMDq)

## Goal
This package aims to make analysis and experimentation of generative AI/ML models more accessible, by providing a simple and consistent interface to foundation models for Data Scientists, Machine Learning Engineers and Software Developers. It's a work in progress, so very much open for collaboration and contribution. 
<br><br>
**Current supported generative AI/ML category** <br>
*Language models (fine tuning, prompt engineering, prompt tuning, model evaluation)*
<br><br>
**Current supported foundation models** <br>
*[HuggingFace Hub](https://huggingface.co) - open source collections of GPT-2, FLAN-T5, EleutherAI, Cerebras, StabilityAI, H2O, Salesforce language models* <br>
*[OpenAI](https://openai.com) - text-davinci-002, text-davinci-003 (GPT3/3.5 base completions model) language models*
<br><br>
**Current supported evals** <br>
*Coming later...*
<br>

## Installation
```
git clone https://github.com/wanoz/panml.git
```

## Usage
### Importing the module
```
# Import panml
from panml.models import ModelPack

# Import other modules/packages as required
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
...
```

### Using open source models from HuggingFace Hub
Create model pack to load model from HuggingFace Hub
```
lm = ModelPack(model='gpt2', source='huggingface')
```

Generate output
```
output = lm.predict('hello world is')
print(output['text'])
```
```
# Output
'hello world is a place where people can live and work together, and where people can live and work together, and where people can live and work together'
```

Show probability of output token
```
output = lm.predict('hello world is', display_probability=True)
print(output['probability'][:5]) # show probability of first 5 tokens in the generated output that follows the provided context
```
```
# Output
[{'token': ' a', 'probability': 0.052747420966625214},
 {'token': ' place', 'probability': 0.045980263501405716},
 {'token': ' where', 'probability': 0.4814596474170685},
 {'token': ' people', 'probability': 0.27657589316368103},
 {'token': ' can', 'probability': 0.2809840738773346}]
```
 
Fine tune the model with your own data from Pandas dataframe - execute in self-supervised autoregressive training regime.
```
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

Generate output with the fine tuned model
```
output = lm.predict('hello world is', display_probability=True)
print(output['text'])
```

Load the locally fine tuned model for use
```
new_lm = ModelPack(model='./results/model_my_tuned_gpt2/', source='local')
```

### Using models from OpenAI
Create model pack from OpenAI model description and API key
```
lm = ModelPack(model='text-davinci-002', source='openai', api_key=<your_openai_key>)
```

Generate output
```
output = lm.predict('What is the best way to live a healthy lifestyle?')
output['text']
```
```
# Output
\nThe best way to live a healthy lifestyle is to eat healthy foods, get regular exercise, 
and get enough sleep.
```

Show probability of output token
```
output = lm.predict('What is the best way to live a healthy lifestyle?', display_probability=True)
print(output['probability'][:5]) # show probability of first 5 tokens in the generated output that follows the provided context
```
```
# Output
[{'token': '\n', 'probability': 0.9912449516093955},
 {'token': 'The', 'probability': 0.40432789860673046},
 {'token': ' best', 'probability': 0.9558591494467851},
 {'token': ' way', 'probability': 0.9988543268851316},
 {'token': ' to', 'probability': 0.9993104225678759}]
```

Generate output using multi-stage prompting (via a prompt modifier) <br>
```
prompts = [
    {'prepend': 'you are a sports coach'},
    {'prepend': 'produce a daily exercise plan for one week'},
    {'prepend': 'summarise to the original question'},
]

output = lm.predict('What is the best way to live a healthy lifestyle?', prompt_modifier=prompts, max_tokens=600)
output['text']
```
###### *Note: Prompt modifier is a list where each item specifies text to be prepended (attached before) and/or appended (attched after) the query/prompt. For example, the texts in first item of list will be prepended/appended to the initial query/prompt, and the texts of the second item in the list will be prepended/appended to the returned LLM response, and the resulting follow-up query/prompt will then be automatically issued to the LLM. This repeats recursively through all of the prompts in the prompt modifier list*

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

Generate embedding
```
output = lm.embedding('What is the best way to live a healthy lifestyle?')
print(output[:5]) # show first 5 embedding elements
```
```
# Output
[0.025805970653891563,
 0.007422071415930986,
 0.01738160289824009,
 -0.006787706166505814,
 -0.003324073040857911]
```

Autogenerate and execute code
```
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

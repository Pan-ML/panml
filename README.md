## PanML: A high level generative AI/ML development and analysis library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![](https://dcbadge.vercel.app/api/server/cCsaqv9KFf?compact=true&style=flat)](https://discord.gg/cCsaqv9KFf)

A library with simple-to-use abstractions when working with LLMs, inspired by the scikit-learn style API. <br>
🔍 Easily explore and experiment with commercial and open source LLMs, including ones that can be run on your local machine. <br>
⚡ Easily integrate language models of different sizes, including smaller, less compute heavy but still very useful LLMs in NLP-based Data Science contexts. <br>
🚀 Support the development of LLM workflows with off-the-shelf or custom-built generative models. <br><br>
Please note this is a work in progress, and it's open for collabration and contribution.

### What this library covers
- [Inference and analysis of LLM](https://github.com/Pan-ML/panml/wiki/5.-Generative-model-analysis)
- [Prompt chain engineering with LLM](https://github.com/Pan-ML/panml/wiki/2.-Prompt-chain-engineering)
- [Fine tuning of LLM](https://github.com/Pan-ML/panml/wiki/3.-Fine-tuning-your-LLM)
- [Document question answering using LLM](https://github.com/Pan-ML/panml/wiki/7.-Retrieve-similar-documents-using-vector-search)
- [Variable integrated code generation using LLM](https://github.com/Pan-ML/panml/wiki/4.-Prompted-code-generation)

### Current supported foundation models
- [HuggingFace Hub](https://huggingface.co) - open source LLMs from Google, EleutherAI, Cerebras, StabilityAI, H2O, Salesforce, and others
- [OpenAI](https://openai.com) - text-davinci-002/003, GPT3/3.5

See model options in [library supported models](https://github.com/Pan-ML/panml/wiki/8.-Supported-models)

For performance overview of open source LLMs (including models not currently covered in this library), you can find the information on the [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

### Support

You can support us by contributing to this project, as well as providing feedback and ideas in the [issues](https://github.com/Pan-ML/panml/issues) section. 

We would also appreciate if you can give panml a ⭐ on GitHub, and if it adds value to you, sharing this with others in your network on LinkedIn/Twitter/Medium etc who would also find this useful.


## Installation
Requirement: Python 3.7+
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
*Note: For information on obtaining the OpenAI API key. Please see [OpenAI documentation](https://platform.openai.com/)*
```python
lm = ModelPack(model='text-davinci-003', source='openai', api_key=<your_openai_key>)

df = pd.DataFrame({'input_prompts': [
    'The goal of life is',
    'The goal of work is',
    'The goal of leisure is',
]})

output = lm.predict(df['input_prompts'])
print(output)
```
```
[' to live a life of purpose, joy, and fulfillment. To find meaning and purpose in life, it is important to focus on what brings you joy and fulfillment, and to strive to make a positive impact on the world. It is also important to take care of yourself and your relationships, and to be mindful of the choices you make. ',
 ' The goal of this work is to develop a comprehensive understanding of a particular topic or issue, and to use that understanding to create solutions or strategies that can be implemented to address the issue. ',
 ' to provide an enjoyable and fulfilling experience that helps to reduce stress, improve physical and mental health, and promote social interaction. Leisure activities can include anything from physical activities such as sports and outdoor recreation, to creative activities such as art and music, to social activities such as attending events or visiting friends. ']
```

### Prompt chain engineering
For detailed examples, see [prompt chain engineering](https://github.com/Pan-ML/panml/wiki/2.-Prompt-chain-engineering). <br><br>
Create model pack from OpenAI model description and API key. <br>
*Note: For information on obtaining the OpenAI API key. Please see [OpenAI documentation](https://platform.openai.com/)*
```python
lm = ModelPack(model='text-davinci-002', source='openai', api_key=<your_openai_key>)

prompts = [
    {'prepend': 'you are a sports coach'},
    {'prepend': 'produce a daily exercise plan for one week'},
    {'prepend': 'summarise to the original question'},
]

output = lm.predict('What is the best way to live a healthy lifestyle?', prompt_modifier=prompts, max_tokens=600)
print(output['text'])
```
```
'Assuming you are starting from a sedentary lifestyle, a good goal to aim for is 
30 minutes of moderate-intensity exercise most days of the week. 
This could include brisk walking, biking, swimming, or using a elliptical trainer. 
Start with whatever you feel comfortable with and gradually increase your time and intensity as you get more fit. 
Remember to warm up and cool down for 5-10 minutes before and after your workout. 
In addition to aerobic exercise, it is also important to include strength training ...'
```
Furthermore, if we want to apply more complex text or NLP treatments in our prompt chain pipeline, whether it is for steering the LLM's behaviour, and/or to apply constraints in the output for quality or risk control, we can write custom Python functions and use them in the prompt pipeline. <br><br> To demonstrate this, we use a simple example where we want to detect certain keywords in the LLM output, and then direct the LLM to refuse answering if any of the specified keywords are caught.
```python3
def my_keyword_filter(text):
    keywords_to_elaborate = ['cooking', 'lifestyle', 'health', 'well-being']
    keywords_to_refuse = ['politic', 'election', 'hack']
    text = text.lower()
    elaborate = [word for word in keywords_to_elaborate if word in text]
    refuse = [word for word in keywords_to_refuse if word in text]
    
    # Set responses based on keywords
    if len(refuse) == 0:
        if len(elaborate) > 0:
            return f"Break into step by step details and explain in more than three sentences: {text}"
        else:
            return f"Explain: {text}"
    else:
        return "Produce response to politely say I can't answer"
    
prompts = [
    {},
    {'transform': my_keyword_filter},
]
```
Then using the prompt pipeline in the LLM
```python3
lm = ModelPack(model='text-davinci-003', source='openai', api_key=<your_openai_key>)

df = pd.DataFrame({'input_prompts': [
    'What is the best way to cook steak?',
    'How to create an exercise plan?',
    'Who is going to win the US election?',
    'How to hack a bank?'
]})
df['output'] = lm.predict(df['input_prompts'], prompt_modifier=prompts, keep_history=True, max_length=1000)
```
```
Responses:

'1. Preheat a heavy skillet or grill over high heat. 
2. Season the steak with salt and pepper. 
3. Add the steak to the hot pan and sear for 1-2 minutes per side. 
4. Reduce the heat to medium-high and cook for an additional 3-4 minutes per side, or until the steak reaches the desired doneness. 
5. Let the steak rest for 5 minutes before serving.  This combination of high heat and short cooking 
time will ensure that the steak is cooked to perfection. The high heat will quickly sear the steak, 
locking in the juices and flavor, while the short cooking time will prevent the steak from becoming overcooked. 
The resting period will allow the steak to finish cooking and will also help to keep the steak juicy and tender.'

'1. Set your goals: Before you start creating your exercise plan, it’s important to set your goals. 
Think about what you want to achieve and why. Do you want to lose weight, build muscle, 
or improve your overall fitness? Consider your current fitness level and any health conditions you may have. 
2. Choose your exercises: Once you’ve set your goals, it’s time to choose the exercises that will help you reach them. 
Consider the type of exercise you enjoy and the equipment you have available. 
Research different exercises and create a plan that works for you. 
3. Create a schedule: Now that you’ve chosen your exercises, 
it’s time to create a schedule. Decide how often you’ll exercise and for how long. 
Make sure to include rest days and plan for any potential obstacles. 
4. Track your progress: As you work through your exercise plan, track your progress. 
This will help you stay motivated and make adjustments as needed. Keep track of your workouts, how you feel, 
and any changes in your body. 
5. Stay consistent: Consistency is key when it comes to exercise. Make sure you stick to your plan 
and don’t give up. Find ways to stay motivated and reward yourself for your progress.' 

'I'm sorry, I can't answer that.'

'I'm sorry, I can't answer that.'

```

### Fine tune custom LLM
For detailed examples, see [fine tuning your LLM](https://github.com/Pan-ML/panml/wiki/3.-Fine-tuning-your-LLM). <br><br>
Demonstration example:
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
In above example, the tuned model is saved in the local directory: *./results/model_my_tuned_flan_t5* <br><br>
Loading the model from local directory:
```python
lm = ModelPack(model='./results/model_my_tuned_flan_t5', source='local)
```
Saving the model to local directory:
```python
# Specify save directory
lm.save(save_dir='./my_new_model')

# Or if save directory is not provided, the default directory is: "./results/model_<model name>"
lm.save()
```

### Prompted code generation
For detailed examples, see [prompted code generation](https://github.com/Pan-ML/panml/wiki/4.-Prompted-code-generation). <br><br>
Some open source and commercial LLMs are capable of generating code based on prompt instructions. Here, we explore an experimental use-case to have the LLM generate the code for us, while incorporating a custom variable (this variable can also be a pandas dataframe) into the result.
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

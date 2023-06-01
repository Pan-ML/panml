"""
Constants used in the PanML package
"""

# Supported LLMs for generation
SUPPORTED_LLMS = {
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
        'togethercomputer/GPT-JT-6B-v1',
        'togethercomputer/GPT-NeoXT-Chat-Base-20B',
        'StabilityAI/stablelm-base-alpha-3b',
        'StabilityAI/stablelm-base-alpha-7b',
        'StabilityAI/stablelm-tuned-alpha-3b',
        'StabilityAI/stablelm-tuned-alpha-7b',
        'h2oai/h2ogpt-oasst1-512-12b',
        'h2oai/h2ogpt-oasst1-512-20b',
        'Salesforce/codegen-350M-multi',
        'Salesforce/codegen-2B-multi',
        'bigcode/starcoder',
        'bert-base-uncased',
        'distilbert-base-uncased',
        'roberta-base',
        'distilroberta-base',
        'togethercomputer/RedPajama-INCITE-Base-3B-v1',
        'togethercomputer/RedPajama-INCITE-Instruct-3B-v1',
        'togethercomputer/RedPajama-INCITE-Chat-3B-v1',
        'tiiuae/falcon-7b',
        'tiiuae/falcon-7b-instruct',
        'tiiuae/falcon-40b',
        'tiiuae/falcon-40b-instruct',
    ],
    'openai': [
        'text-davinci-002', 
        'text-davinci-003',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0301',
    ],
}

# Supported OpenAI models for code completion
SUPPORTED_OAI_CODE_MODELS = [
    'text-davinci-002', 
    'text-davinci-002',
]

# Supported OpenAI models for completion
SUPPORTED_OAI_COMPLETION_MODELS = [
    'text-davinci-002', 
    'text-davinci-003',
]

# Supported OpenAI models for chat
SUPPORTED_OAI_CHAT_MODELS = [
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-0301',
]

# Supported sources
SUPPORTED_LLM_SOURCES = [
    'huggingface', 
    'local', 
    'openai',
]

# Supported vector search sources
SUPPORTED_VECTOR_SEARCH_SOURCES = [
    'faiss', 
    'pinecone',
]

# Supported embedding models
SUPPORTED_EMBEDDING_MODELS = {
    'huggingface': [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
        'all-distilroberta-v1'
        'nq-distilbert-base-v1',
        'paraphrase-albert-small-v2',
        'paraphrase-MiniLM-L3-v2',
        'paraphrase-MiniLM-L6-v2',
        'multi-qa-MiniLM-L6-cos-v1',
        'multi-qa-distilbert-cos-v1',
        'msmarco-MiniLM-L6-cos-v5',
        'distiluse-base-multilingual-cased-v1',
        'distiluse-base-multilingual-cased-v2',
        'paraphrase-multilingual-MiniLM-L12-v2',
        'paraphrase-multilingual-mpnet-base-v2',
        'distilbert-base-nli-stsb-mean-tokens',
    ],
    'openai': [
        'text-embedding-ada-002', 
    ],
}

# Prompt templates
CHAIN_OF_THOUGHT_PROMPT = [
    {"append": "Let's think this step by step."}
]

EXPLAIN_PROMPT = [
    {"append": "Explain and answer."}
]

SIMPLIFY_PROMPT = [
    {"append": "Explain to me like I'm a 6 year old."}
]

SUMMARIZE_PROMPT = [
    {"prepend": "Summarise the following: "}
]

QUESTION_ANSWER_PROMPT = [
    {"append": ""},
    {"append": "Break down into details."},
    {"append": "Summarise to the address original question."}
]

INTERROGATE_PROMPT = [
    {"append": ""},
    {"append": "Explain why this is true: "},
    {"append": "Explain why this is true: "},
]
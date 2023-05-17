from setuptools import setup, find_packages

setup(
  name = 'panml',   
  packages = find_packages(),
  version = '0.0.9',
  license = 'MIT',
  description = 'PanML is a generative AI/ML development toolkit designed for ease of use and fast experimentation.',
  long_description = 'PanML aims to make analysis and experimentation of generative AI/ML models more accessible, by providing a simple and consistent interface to foundation models for Data Scientists, Machine Learning Engineers and Software Developers. \
                      The package provides various assistive tools to work with generative language models, such as prompt engineering, fine tuning and others. \
                      \n\nQuick start guide: https://github.com/Pan-ML/panml/wiki/1.-Quick-start-guide \
                      \n\nDocumentation/Wiki: https://github.com/Pan-ML/panml/wiki',
  author = 'PanML team',
  author_email = 'williamxn.z@gmail.com',
  url = 'https://github.com/Pan-ML/panml',
  keywords = [
        'generative AI', 
        'generative model', 
        'machine learning', 
        'large language model',
        'LLM', 
        'prompt engineering', 
        'fine tuning', 
        'prompt tuning', 
        'retrieval augmentation', 
        'AI safety',
        'AI alignment'
        ], 
  install_requires=[
        'huggingface-hub>=0.14.1',
        'numpy>=1.24.3',
        'openai>=0.27.6',
        'pandas>=2.0.1',
        'tokenizers>=0.13.3',
        'datasets>=2.12.0',
        'torch>=2.0.0',
        'tqdm>=4.65.0',
        'transformers>=4.29.1',
        'faiss-cpu>=1.7.4',
        'sentence-transformers>=2.2.2',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
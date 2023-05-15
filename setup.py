
from distutils.core import setup
setup(
  name = 'panml', # package name     
  packages = ['panml'], # package name
  version = '0.0.1', # version
  license='MIT', # license
  description = 'PanML is a lightweight generative AI/ML development toolkit designed for ease of use and fast experimentation.', # short description about the package
  author = 'PanML team', # team name
  author_email = 'williamxn.zl@gmail.com', # contact email
  url = 'https://github.com/Pan-ML/panml', # url link
  download_url = 'https://github.com/Pan-ML/panml/archive/refs/tags/v0.0.1.tar.gz', # set a release on GitHub (with release tag) -> Assets -> right click and copy the link
  keywords = ['generative AI', 'generative model', 'machine learning', 'large language model', # keywords
              'LLM', 'prompt engineering', 'fine tuning', 'prompt tuning', 'retrieval augmentation', 
              'AI safety', 'AI alignment'], 
  install_requires=[  # dependencies
        'aiohttp>=3.8.4',
        'aiosignal>=1.3.1',
        'async-timeout>=4.0.2',
        'attrs>=23.1.0',
        'certifi>=2023.5.7',
        'charset-normalizer>=3.1.0',
        'datasets>=2.12.0',
        'dill>=0.3.6',
        'filelock>=3.12.0',
        'frozenlist>=1.3.3',
        'fsspec>=2023.5.0',
        'huggingface-hub>=0.14.1',
        'idna>=3.4',
        'Jinja2>=3.1.2',
        'MarkupSafe>=2.1.2',
        'mpmath>=1.3.0',
        'multidict>=6.0.4',
        'multiprocess>=0.70.14',
        'networkx>=3.1',
        'numpy>=1.24.3',
        'openai>=0.27.6',
        'packaging>=23.1',
        'pandas>=2.0.1',
        'pyarrow>=12.0.0',
        'python-dateutil>=2.8.2',
        'pytz>=2023.3',
        'PyYAML>=6.0',
        'regex>=2023.5.5',
        'requests>=2.30.0',
        'responses>=0.18.0',
        'six>=1.16.0',
        'sympy>=1.12',
        'tokenizers>=0.13.3',
        'torch>=2.0.1',
        'tqdm>=4.65.0',
        'transformers>=4.29.1',
        'typing_extensions>=4.5.0',
        'tzdata>=2023.3',
        'urllib3>=2.0.2',
        'xxhash>=3.2.0',
        'yarl>=1.9.2',
        'faiss-cpu>=1.7.4',
        'faiss-gpu>=1.7.2',
        'sentence-transformers>=2.2.2',
      ],
  classifiers=[
    'Development Status :: 3 - Beta', # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of the package
    'Intended Audience :: Developers', # Define the audience type
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License', # license
    'Programming Language :: Python :: 3', # Specify supported Python versions
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
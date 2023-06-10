from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
  name = 'panml', # package name     
  packages = find_packages(exclude=['test']), # package name
  version = '1.0.1', # version
  license = 'MIT', # license
  description = 'PanML is a high level generative AI/ML development library designed for ease of use and fast experimentation.', # short description about the package
  long_description=long_description,
  long_description_content_type='text/markdown',
  author = 'PanML team', # team name
  author_email = 'teampanml@gmail.com', # contact email
  url = 'https://github.com/Pan-ML/panml', # url link
  # download_url = 'https://github.com/Pan-ML/panml/archive/refs/tags/v0.0.9.tar.gz', # set a release on GitHub (with release tag) -> Assets -> right click and copy the link
  keywords = ['generative AI', 'generative model', 'machine learning', 'large language model', # keywords
              'LLM', 'prompt engineering', 'fine tuning', 'prompt tuning', 'retrieval augmentation', 
              'AI safety', 'AI alignment'], 
  install_requires=[  # dependencies
        'huggingface-hub>=0.14.1',
        'numpy>=1.22.4',
        'openai>=0.27.6',
        'pandas>=2.0.1',
        'tokenizers>=0.13.3',
        'datasets>=2.12.0',
        'torch>=2.0.0',
        'tqdm>=4.65.0',
        'transformers<=4.27.4',
        'faiss-cpu>=1.7.4',
        'sentence-transformers>=2.2.2',
        'accelerate<=0.19.0',
        'peft==0.3.0',
      ],
  classifiers=[
    'Development Status :: 4 - Beta', # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of the package
    'Intended Audience :: Science/Research', # Define the audience type
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Information Technology',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License', # License
    'Programming Language :: Python :: 3', # Supported Python versions
    'Programming Language :: Python :: 3.7', 
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
  ],
)
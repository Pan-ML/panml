from distutils.core import setup

__version__ = '0.0.9'

setup(
  name = 'panml', # package name     
  packages = ['panml'], # package name
  version = __version__, # version
  license = 'MIT', # license
  description = 'PanML is a generative AI/ML development toolkit designed for ease of use and fast experimentation.', # short description about the package
  long_description = 'PanML aims to make analysis and experimentation of generative AI/ML models more accessible, by providing a simple and consistent interface to foundation models for Data Scientists, Machine Learning Engineers and Software Developers. \
                      The package provides various assistive tools to work with generative language models, such as prompt engineering, fine tuning and others. \
                      \n\nQuick start guide: https://github.com/Pan-ML/panml/wiki/1.-Quick-start-guide \
                      \n\nDocumentation/Wiki: https://github.com/Pan-ML/panml/wiki', # long description
  author = 'PanML team', # team name
  author_email = 'williamxn.z@gmail.com', # contact email
  url = 'https://github.com/Pan-ML/panml', # url link
  download_url = 'https://github.com/Pan-ML/panml/archive/refs/tags/vX.X.X.tar.gz', # set a release on GitHub (with release tag) -> Assets -> right click and copy the link
  keywords = ['generative AI', 'generative model', 'machine learning', 'large language model', # keywords
              'LLM', 'prompt engineering', 'fine tuning', 'prompt tuning', 'retrieval augmentation', 
              'AI safety', 'AI alignment'], 
  install_requires=[  # dependencies
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
    'Development Status :: 4 - Beta', # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of the package
    'Intended Audience :: Developers', # Define the audience type
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License', # license
    'Programming Language :: Python :: 3', # Specify supported Python versions
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
import unittest
from panml.search import VectorEngine
from test.corpus import TEST_CORPUS_1
class TestVectorEngine(unittest.TestCase):
    '''
    Run tests on VecterEngine class in search.py
    '''

    # Test case: handle invalid source input
    print('Setup test_faiss_hf_invalid_source_input')
    def test_faiss_hf_invalid_source_input(self):
        # test invalid source as int
        with self.assertRaises(ValueError):
            m = VectorEngine(model=None, source=1)
        # test invalid source as float
        with self.assertRaises(ValueError):
            m = VectorEngine(model=None, source=0.2)
        # test invalid source as non accepted str
        with self.assertRaises(ValueError):
            m = VectorEngine(model=None, source='faiss1')

    # Test case: handle invalid model input
    print('Setup test_faiss_hf_invalid_model_input')
    def test_faiss_hf_invalid_model_input(self):
        # test invalid model as int
        with self.assertRaises(ValueError):
            m = VectorEngine(model=1, source='faiss')
        # test invalid model as float
        with self.assertRaises(ValueError):
            m = VectorEngine(model=0.2, source='faiss')
        # test invalid model as non accepted str
        with self.assertRaises(ValueError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens1', source='faiss')

    # Test case: handle invalid corpus input
    print('Setup test_faiss_hf_invalid_corpus_input_storage')
    def test_faiss_hf_invalid_corpus_input_storage(self):
        # test invalid corpus store input type
        with self.assertRaises(TypeError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            m.store(1, model_name=None)

        # test invalid corpus store input type
        with self.assertRaises(TypeError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            m.store(0.2, model_name=None)

        # test invalid corpus store input type - None
        with self.assertRaises(TypeError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            m.store(None, model_name=None)

        # test invalid corpus store input value - empty
        with self.assertRaises(ValueError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            TEST_CORPUS = []
            m.store([], model_name=None)

    # Test case: handle invalid model input
    print('Setup test_faiss_hf_invalid_corpus_input_query')
    def test_faiss_hf_invalid_corpus_input_query(self):
        # test invalid corpus query input type
        with self.assertRaises(TypeError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            TEST_SAMPLE = 1
            m.store(TEST_CORPUS_1, model_name=None)
            output = m.search(TEST_SAMPLE, 3)

        # test invalid corpus query input type
        with self.assertRaises(TypeError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            TEST_SAMPLE = 0.2
            m.store(TEST_CORPUS_1, model_name=None)
            output = m.search(TEST_SAMPLE, 3)

        # test invalid corpus query input type - None
        with self.assertRaises(TypeError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            TEST_SAMPLE = None
            m.store(TEST_CORPUS_1, model_name=None)
            output = m.search(TEST_SAMPLE, 3)

        # test invalid corpus query input type
        with self.assertRaises(TypeError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            TEST_SAMPLE = []
            m.store(TEST_CORPUS_1, model_name=None)
            output = m.search(TEST_SAMPLE, 3)

        # test invalid corpus query input is empty
        with self.assertRaises(ValueError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            TEST_SAMPLE = ''
            m.store(TEST_CORPUS_1, model_name=None)
            output = m.search(TEST_SAMPLE, 3)

        # test invalid corpus k input type
        with self.assertRaises(TypeError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            TEST_SAMPLE = 'white rabbit'
            m.store(TEST_CORPUS_1, model_name=None)
            output = m.search(TEST_SAMPLE, 0.1)

        # test invalid corpus k input is less than 1
        with self.assertRaises(ValueError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens', source='faiss')
            TEST_SAMPLE = 'white rabbit'
            m.store(TEST_CORPUS_1, model_name=None)
            output = m.search(TEST_SAMPLE, -1)

    # Test case: validate FAISS vector search functionality
    print('Setup test_faiss_hf_vector_search')
    def test_faiss_hf_vector_search(self):
        # test sample input
        TEST_SAMPLES = [
            {
                'text': 'white rabbit',
                'k': 3,
                'expected_output': [
                    'The slow white rabbit is lazy',
                    'The quick brown rabbit jumps over the lazy dog',
                    'The quick brown rabbit jumps over the lazy cat'
                ],
            },
        ]
        # Run vector search validation
        m = VectorEngine(source='faiss')
        m.store(TEST_CORPUS_1, model_name=None)
        for sample in TEST_SAMPLES:
            output = m.search(sample['text'], sample['k']) # get test output
            self.assertEqual(output, sample['expected_output'], 'FAISS vector search output is not as expected, review may be required.') # check test output against expected output

    # Test case: validate FAISS vector search missing OpenAI key
    print('Setup test_faiss_missing_openai_key')
    def test_faiss_missing_openai_key(self):
        # test no openai api key provided
        with self.assertRaises(ValueError):
            m = VectorEngine(model='text-davinci-002', source='faiss')

    # Test case: validate FAISS vector store save functionality
    print('Setup test_faiss_invalid_model_name_vector_store')
    def test_faiss_invalid_model_name_vector_store(self):
        # test invalid model name input type
        with self.assertRaises(TypeError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, model_name=1)

    # Test case: validate FAISS vector store load vectors functionality
    print('Setup test_faiss_invalid_load_vectors_dir')
    def test_faiss_invalid_load_vectors_dir(self):
        # test invalid vectors dir input type
        with self.assertRaises(TypeError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.load(vectors_dir=1, corpus_dir='corpus.pkl')

        with self.assertRaises(TypeError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.load(vectors_dir=None, corpus_dir='corpus.pkl')

        # test invalid vectors dir input name
        with self.assertRaises(ValueError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.load(vectors_dir='vectors', corpus_dir='corpus.pkl')

    # Test case: validate FAISS vector store load corpus functionality
    print('Setup test_faiss_invalid_load_corpus_dir')
    def test_faiss_invalid_load_corpus_dir(self):
        # test invalid load corpus dir type
        with self.assertRaises(TypeError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.load(vectors_dir='vectors.faiss', corpus_dir=1)

        with self.assertRaises(TypeError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.load(vectors_dir='vectors.faiss', corpus_dir=None)

        # test invalid load corpus dir input name
        with self.assertRaises(ValueError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.load(vectors_dir='vectors.faiss', corpus_dir='corpus')

    # Test case: validate FAISS vector store mode and mode related arguments
    print('Setup test_faiss_invalid_mode_arguments')
    def test_faiss_invalid_mode_arguments(self):
        # test invalid input mode type
        with self.assertRaises(TypeError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode=1, model_name=None)

        # test invalid input mode value
        with self.assertRaises(ValueError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='flat1', model_name=None)

        # test invalid input nlist type
        with self.assertRaises(TypeError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='ivfflat', nlist='test', model_name=None)

        # test invalid input nlist value
        with self.assertRaises(ValueError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='ivfflat', nlist=-1, model_name=None)

        # test invalid input m type
        with self.assertRaises(TypeError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='ivfpq', m='test', bits=8, model_name=None)

        # test invalid input m value
        with self.assertRaises(ValueError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='ivfpq', m=-1, bits=8, model_name=None)

        # test invalid input bits type
        with self.assertRaises(TypeError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='ivfpq', m=8, bits='test', model_name=None)

        # test invalid input bits value
        with self.assertRaises(ValueError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='ivfpq', m=8, bits=-1, model_name=None)

        # test missing m for ivfpq
        with self.assertRaises(ValueError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='ivfpq', m=8, model_name=None)

        # test missing bits for ivfpq
        with self.assertRaises(ValueError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='ivfpq', bits=8, model_name=None)

        # test missing both m and q for ivfpq
        with self.assertRaises(ValueError):
            # Run vector search validation
            m = VectorEngine(source='faiss')
            m.store(TEST_CORPUS_1, mode='ivfpq', model_name=None)

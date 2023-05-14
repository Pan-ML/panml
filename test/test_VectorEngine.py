import unittest
from panml.utils import VectorEngine

class TestVectorEngine(unittest.TestCase):
    '''
    Run tests on VecterEngine class in models.py
    '''
    # Test case: handle invalid source input
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

    # Test case: validate FAISS vector search functionality
    def test_faiss_hf_vector_search(self):
        # test sample input
        TEST_CORPUS = [
            'The quick brown fox jumps over the lazy dog', 
            'The quick brown rabbit jumps over the lazy dog', 
            'The quick brown rabbit jumps over the lazy cat', 
            'The quick brown cat jumps over the lazy dog', 
            'The quick brown cat jumps over the lazy rabbit', 
            'The quick brown cat jumps over the lazy fox', 
            'The quick brown cat is lazy', 
            'The quick brown fox is lazy', 
            'The slow brown dog is lazy', 
            'The slow white rabbit is lazy',
        ]
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
        m.store(TEST_CORPUS)
        for sample in TEST_SAMPLES:
            output = m.search(sample['text'], sample['k']) # get test output
            self.assertEqual(output, sample['expected_output'], 'FAISS vector search output is not as expected, review may be required.') # check test output against expected output
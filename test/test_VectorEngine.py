import unittest
from panml.models import VectorEngine

class TestVectorEngine(unittest.TestCase):
    '''
    Run tests on VecterEngine class in models.py
    '''
    # Test case 1: handle invalid source input
    def test_invalid_source_input(self):
        # test invalid source as int
        with self.assertRaises(ValueError):
            m = VectorEngine(model=None, source=1)
        # test invalid source as float
        with self.assertRaises(ValueError):
            m = VectorEngine(model=None, source=0.2)
        # test invalid source as non accepted str
        with self.assertRaises(ValueError):
            m = VectorEngine(model=None, source='faiss1')

    # Test case 2: handle invalid model input
    def test_invalid_source_input(self):
        # test invalid model as int
        with self.assertRaises(ValueError):
            m = VectorEngine(model=1, source='faiss')
        # test invalid model as float
        with self.assertRaises(ValueError):
            m = VectorEngine(model=0.2, source='faiss')
        # test invalid model as non accepted str
        with self.assertRaises(ValueError):
            m = VectorEngine(model='distilbert-base-nli-stsb-mean-tokens1', source='faiss')


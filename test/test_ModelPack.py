import unittest
from panml.models import ModelPack

class TestModelPack(unittest.TestCase):
    '''
    Run tests on ModelPack class in models.py
    '''
    # Test case: handle invalid source input
    print('Setup test_modelpack_invalid_source_input')
    def test_modelpack_invalid_source_input(self):
        # test invalid source as int
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt2', source=1)
        # test invalid source as float
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt2', source=0.2)
        # test invalid source as non accepted str
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt2', source='huggingface1')

    # Test case: handle invalid model input
    print('Setup test_modelpack_invalid_model_input')
    def test_modelpack_invalid_model_input(self):
        # test invalid model as int
        with self.assertRaises(ValueError):
            m = ModelPack(model=1, source='huggingface')
        # test invalid model as float
        with self.assertRaises(ValueError):
            m = ModelPack(model=0.2, source='huggingface')
        # test invalid model as non accepted str
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt99', source='huggingface')

    # Test case: handle model and source mismatch input
    print('Setup test_modelpack_incorrect_model_source_input')
    def test_modelpack_incorrect_model_source_input(self):
        # test invalid model and source match combo 1
        with self.assertRaises(ValueError):
            m = ModelPack(model='text-davinci-002', source='huggingface')
        # test invalid model and source match combo 2
        with self.assertRaises(ValueError):
            m = ModelPack(model='gpt2', source='openai')

    # Test case: handle model and source correct match input
    print('Setup test_modelpack_correct_model_source_gpt2_input')
    def test_modelpack_correct_model_source_gpt2_input(self):
        # test valid model and source match combo 1
        m = ModelPack(model='gpt2', source='huggingface')

    # Test case: handle no OpenAI API key input
    print('Setup test_modelpack_missing_openai_api_key_input')
    def test_modelpack_missing_openai_api_key_input(self):
        # test no openai api key provided
        with self.assertRaises(ValueError):
            m = ModelPack(model='text-davinci-002', source='openai')
    
    # Test case: handle model and source correct match input
    print('Setup test_modelpack_correct_model_source_flan_input')
    def test_modelpack_correct_model_source_flan_input(self):
        # test valid model and source match combo 2
        m = ModelPack(model='google/flan-t5-small', source='huggingface')



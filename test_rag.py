import sys
import os
sys.path.append(r'c:\Users\Funky\Desktop\cs4248\api-inference')
import importlib
api_local = importlib.import_module('api-inference-local')
rag = api_local.MemeCapRAG(r'c:\Users\Funky\Desktop\cs4248\memecap-data\memes-test.json')
print('Valid indices count:', len(rag.valid_data_indices))
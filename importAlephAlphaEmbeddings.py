"""
    Aleph Alpha (luminous) embeddings. This script should be run only once.
"""

import os
import pandas as pd

# Aleph Alpha endpoints
from aleph_alpha_client import Client, Prompt, SemanticRepresentation, SemanticEmbeddingRequest
from dotenv import load_dotenv
load_dotenv()

data = pd.read_csv('scales.csv')
# additional check on the scales converted here
data = data[data['scaleID'].isin(['PID', 'NEO'])]

# if the scales items are language specific, rename the column to 
# 'item' and drop the other language column
if 'item_de' in data.columns:
    data = data.rename(columns={'item_de': 'item'})
    data = data.drop(columns=['item_en'])

# instantiate the Aleph Alpha client
client = Client(token=os.getenv("ALEPHALPHA_API_TOKEN"))

# check if the Aleph Alpha embeddings file exists
if os.path.exists('embeddings_alephalpha_base_128_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file embeddings_alephalpha_base_128_de.csv ' + 
        'already exists. Please remove it before running this script again.')

# get embeddings from Aleph Alpha
symmetric_embeddings = []
for text in data.item.values:
    symmetric_params = {
        "prompt": Prompt.from_text(text),
        "representation": SemanticRepresentation.Symmetric,
        "compress_to_size": 128
    }
    symmetric_request = SemanticEmbeddingRequest(**symmetric_params)
    symmetric_response = client.semantic_embed(request=symmetric_request, model="luminous-base")
    symmetric_embeddings.append(symmetric_response.embedding)

data['embedding'] = symmetric_embeddings
data.to_csv('embeddings_alephalpha_base_128_de.csv', index=False)

# check if the Aleph Alpha embeddings file exists
if os.path.exists('embeddings_alephalpha_base_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file embeddings_alephalpha_base_de.csv ' + 
        'already exists. Please remove it before running this script again.')

symmetric_embeddings = []
for text in data.item.values:
    symmetric_params = {
        "prompt": Prompt.from_text(text),
        "representation": SemanticRepresentation.Symmetric,
        "compress_to_size": None
    }
    symmetric_request = SemanticEmbeddingRequest(**symmetric_params)
    symmetric_response = client.semantic_embed(request=symmetric_request, model="luminous-base")
    symmetric_embeddings.append(symmetric_response.embedding)

data['embedding'] = symmetric_embeddings
data.to_csv('embeddings_alephalpha_base_de.csv', index=False)

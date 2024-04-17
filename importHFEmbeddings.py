"""
    Hugging face embeddings. This script should be run only once.
"""

import os
import pandas as pd
from haystack.components.embedders import HuggingFaceTEITextEmbedder
from haystack.utils import Secret

data = pd.read_csv('scales.csv')
# select rows of data where scaleID is either 'PDS', 'NEO', or 'ADS'
data = data[data['scaleID'].isin(['PID', 'NEO', 'ADS'])]

# retrieve embeddings from roberta from hugging face
text_embedder = HuggingFaceTEITextEmbedder(
    model="sentence-transformers/all-roberta-large-v1", 
    token=Secret.from_token(os.environ['HF_API_KEY'])
)

embeddings = [text_embedder.run(item)['embedding'] for item in data.item]

data['embedding'] = embeddings
data.to_csv('scales_roberta_de.csv', index=False)


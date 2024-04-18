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

# check if the 'embeddings_roberta_de.csv' file exists
if os.path.exists('embeddings_roberta_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file embeddings_roberta_de.csv already exists. Please remove it before running this script again.')


# if there is a column 'polarity', drop column 'polarity' from dataframe data
if 'polarity' in data.columns:
    data = data.drop(columns=['polarity'])
data = data.rename(columns={'item_de': 'item'})
data = data.drop(columns=['item_en'])

# retrieve embeddings from roberta from hugging face
text_embedder = HuggingFaceTEITextEmbedder(
    model="sentence-transformers/all-roberta-large-v1", 
    token=Secret.from_token(os.environ['HF_API_KEY'])
)

embeddings = [text_embedder.run(item)['embedding'] for item in data.item]

# save embeddings to the 'scales_roberta_de.csv' file
data['embedding'] = embeddings
data.to_csv('embeddings_roberta_de.csv', index=False)


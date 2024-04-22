"""
    Hugging face embeddings. This script should be run only once.
"""

import os
import pandas as pd
from haystack.components.embedders import HuggingFaceTEITextEmbedder
from haystack.utils import Secret

data = pd.read_csv('scales.csv', encoding='unicode_escape')

# if the scales items are language specific, rename the column to 
# 'item' and drop the other language column
if 'item_de' in data.columns:
    data = data.rename(columns={'item_de': 'item'})
    data = data.drop(columns=['item_en'])

# check if the 'embeddings_roberta_de.csv' file exists
if os.path.exists('embeddings_roberta_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file embeddings_roberta_de.csv ' + 
        'already exists. Please remove it before running this script again.')

# retrieve embeddings from roberta from hugging face
# https://huggingface.co/sentence-transformers/all-roberta-large-v1
text_embedder = HuggingFaceTEITextEmbedder(
    model="sentence-transformers/all-roberta-large-v1", 
    token=Secret.from_token(os.environ['HF_API_KEY'])
)

embeddings = [text_embedder.run(item)['embedding'] for item in data.item]

# save embeddings to the 'scales_roberta_de.csv' file
data['embedding'] = embeddings
data.to_csv('embeddings_roberta_de.csv', index=False)

# check if the 'embeddings_roberta_de.csv' file exists
if os.path.exists('embeddings_mpnetbasev2_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file embeddings_mpnetbasev2_de.csv ' + 
        'already exists. Please remove it before running this script again.')

# a model recommended for its performance (https://www.sbert.net/docs/pretrained_models.html)
text_embedder = HuggingFaceTEITextEmbedder(
    model="sentence-transformers/all-mpnet-base-v2", 
    token=Secret.from_token(os.environ['HF_API_KEY'])
)

embeddings = [text_embedder.run(item)['embedding'] for item in data.item]

# save embeddings to the 'scales_roberta_de.csv' file
data['embedding'] = embeddings
data.to_csv('embeddings_mpnetbasev2_de.csv', index=False)



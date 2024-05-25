"""
    Cohere embeddings. This script should be run only once.
"""

#!pip import cohere

import os
import pandas as pd
import cohere

# check if the embeddings file exists
if os.path.exists('embeddings_cohere_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file embeddings_cohere_de.csv ' + 
        'already exists. Please remove it before running this script again.')

data = pd.read_csv('scales.csv')
# additional check on the scales converted here
data = data[data['scaleID'].isin(['PID', 'NEO'])]

# if the scales items are language specific, rename the column to 
# 'item' and drop the other language column
if 'item_de' in data.columns:
    data = data.rename(columns={'item_de': 'item'})
    data = data.drop(columns=['item_en'])

# create a Cohere object
api_key = os.environ["COHERE_API_KEY"]
co = cohere.Client(api_key)

# get embeddings
response = co.embed(
    texts=list(data.item), 
    model="embed-multilingual-v3.0", 
    input_type="classification"  # or clustering etc.
)

# collect embeddings from response and save them to the embeddings file
data['embedding'] = response.embeddings
data.to_csv('embeddings_cohere_de.csv', index=False)


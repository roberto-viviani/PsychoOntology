"""
    Mistral embeddings. This script should be run only once.
"""
import os
import pandas as pd
from mistralai.client import MistralClient

api_key = os.environ["MISTRAL_API_KEY"]
embed_model = "mistral-embed"

# check if the 'scales_mistral_de.csv' file exists
if os.path.exists('embeddings_mistral_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file embeddings_mistral_de.csv ' + 
        'already exists. Please remove it before running this script again.')

data = pd.read_csv('scales.csv')

# if the scales items are language specific, rename the column to 
# 'item' and drop the other language column
if 'item_de' in data.columns:
    data = data.rename(columns={'item_de': 'item'})
    data = data.drop(columns=['item_en'])

# create a MistralClient object
client = MistralClient(api_key=api_key)

# get embeddings from Mistral
response = client.embeddings(
    model=embed_model,
    input=list(data.item),
)

# collecct embeddings from response and save them to the 'scales_mistral_de.csv' file
data['embedding'] = [x.embedding for x in response.data]
data.to_csv('embeddings_mistral_de.csv', index=False)


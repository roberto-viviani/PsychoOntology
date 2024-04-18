"""
    Mistral embeddings. This script should be run only once.
"""
import os
import pandas as pd
from mistralai.client import MistralClient

api_key = os.environ["MISTRAL_API_KEY"]
embed_model = "mistral-embed"

data = pd.read_csv('scales.csv', encoding='unicode_escape')
# select rows of data where scaleID is either 'PDS', 'NEO', or 'ADS'
data = data[data['scaleID'].isin(['PID', 'NEO', 'ADS'])]

# if there is a column 'polarity', drop column 'polarity' from dataframe data
if 'polarity' in data.columns:
    data = data.drop(columns=['polarity'])
data = data.rename(columns={'item_de': 'item'})
data = data.drop(columns=['item_en'])

# check if the 'scales_mistral_de.csv' file exists
if os.path.exists('embeddings_mistral_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file embeddings_mistral_de.csv already exists. Please remove it before running this script again.')

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


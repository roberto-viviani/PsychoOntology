"""
    Mistral embeddings. This script should be run only once.
"""
import os
import pandas as pd
from mistralai.client import MistralClient

api_key = os.environ["MISTRAL_API_KEY"]
embed_model = "mistral-embed"

data = pd.read_csv('scales.csv')
# select rows of data where scaleID is either 'PDS', 'NEO', or 'ADS'
data = data[data['scaleID'].isin(['PID', 'NEO', 'ADS'])]

client = MistralClient(api_key=api_key)

response = client.embeddings(
    model=embed_model,
    input=list(data.item),
)

# check if the 'scales_mistral_de.csv' file exists
if os.path.exists('scales_mistral_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file scales_mistral_de.csv already exists. Please remove it before running this script again.')

# response.data is a list of dictionaries with a field embedding, which contains a vector.
data['embedding'] = [x.embedding for x in response.data]
data.to_csv('scales_mistral_de.csv', index=False)


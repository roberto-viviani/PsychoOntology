"""
    Imports OpenAI embeddings into the scales.csv file. This function should be run only once.
"""
import pandas as pd

def openAiEmbeddings(items, model_name="text-embedding-3-small"):
    """Retrieves embeddings from openAI"""

    #retrieve embeddings from openAI
    from openai import OpenAI
    client = OpenAI()

    def get_embedding(text, model=model_name):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    # call get_embedding for each item in items using a list comprehension
    embeddings = [get_embedding(item) for item in items]

    return embeddings


data = pd.read_csv('scales.csv', encoding='unicode_escape')
data = data[data['scaleID'].isin(['PID', 'NEO', 'ADS'])]

# if there is a column 'polarity', drop column 'polarity' from dataframe data
if 'polarity' in data.columns:
    data = data.drop(columns=['polarity'])

# retrieve embeddings from openAI small model
data['embedding'] = openAiEmbeddings(data['item_de'], "text-embedding-3-small")
data.rename(columns={'item_de': 'item'}).drop(columns=['item_en']).to_csv('embeddings_openAI_small_de.csv', index=False)

# retrieve embeddings from openAI large model
data['embedding'] = openAiEmbeddings(data['item_de'], "text-embedding-3-large")
data.rename(columns={'item_de': 'item'}).drop(columns=['item_en']).to_csv('embeddings_openAI_large_de.csv', index=False)

# retrieve embeddings from openAI large model English version
data['embedding'] = openAiEmbeddings(data['item_en'], "text-embedding-3-large")
data.rename(columns={'item_en': 'item'}).drop(columns=['item_de']).to_csv('embeddings_openAI_large_en.csv', index=False)




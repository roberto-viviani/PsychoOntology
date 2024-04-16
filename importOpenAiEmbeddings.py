"""
    Imports OpenAI emebddings into the scales.csv file. This function should be run only once.
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


data = pd.read_csv('scales.csv')

# retrieve embeddings from openAI small model
data['embedding'] = openAiEmbeddings(data['item'], "text-embedding-3-small")
data.to_csv('scales_openAI_small.csv', index=False)

# retrieve embeddings from openAI large model
data['embedding'] = openAiEmbeddings(data['item'], "text-embedding-3-large")
data.to_csv('scales_openAI_large.csv', index=False)




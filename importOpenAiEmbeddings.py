"""
    Imports OpenAI emebddings into the scales.csv file. This function should be run only once.
"""
import pandas as pd

def openAiEmbeddings(items):
    """Retrieves embeddings from openAI"""

    #retrieve embeddings from openAI
    from openai import OpenAI
    client = OpenAI()

    def get_embedding(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    # call get_embedding for each item in items using a list comprehension
    embeddings = [get_embedding(item) for item in items]

    return embeddings


data = pd.read_csv('scales.csv')
data['embedding'] = openAiEmbeddings(data['item'])
data.to_csv('scales_openAI.csv', index=False)




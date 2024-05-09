"""
    Imports Universal Sentence Encoder embeddings. This function should be run only once.
"""
import pandas as pd
import tensorflow_hub as hub
import os

# check if the 'scales_mistral_de.csv' file exists
if os.path.exists('embeddings_univSentEncoder_de.csv'):
    # raise error if the file exists
    raise FileExistsError('The file embeddings_univSentEncoder_de.csv ' + 
        'already exists. Please remove it before running this script again.')

data = pd.read_csv('scales.csv')

# if the scales items are language specific, rename the column to 
# 'item' and drop the other language column
if 'item_de' in data.columns:
    data = data.rename(columns={'item_de': 'item'})
    data = data.drop(columns=['item_en'])

# load the model
# Use this if encoding English scale items
#embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")

# The multilingual version of the encoder requires this line and tensorflow-text.
# tensor-flow text is known to work with the encoder in version 2.8.x, that is code 
# !pip install -q -U "tensorflow-text==2.8.*" 
# However, tensorflow-text requires a tensorflow of the same version (possibly requiring
# re-installing), AND there is no tensorflow-text for windows. Best option to use the colab 
# or any other cloud, do the installs, and delete the runtime.
import tensorflow_text
embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/multilingual/2")

# get embeddings
embeddings = embed(list(data.item)).numpy()

# collect embeddings from response and save them to the 'embeddigs.csv' file
data['embedding'] = [list(x) for x in embeddings]
data.to_csv('embeddings_univSentEncoder_de.csv', index=False)



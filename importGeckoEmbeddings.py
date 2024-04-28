"""
    Imports gecko embeddings. This function should be run after loading it to Google's
    colab and only once.
"""

from typing import List
import pandas as pd

from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

def embed_text(
    texts: List[str],
    model_name: str = "textembedding-gecko-multilingual@001",
    task: str = "SEMANTIC_SIMILARITY"
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model."""
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    embeddings = model.get_embeddings(inputs)
    return [embedding.values for embedding in embeddings]

data = pd.read_csv('scales.csv')
# additional check on the scales converted here
data = data[data['scaleID'].isin(['PID', 'NEO'])]

# retrieve embeddings from gecko model
data['embedding'] = embed_text(data['item_de'])
data.rename(columns={'item_de': 'item'}).drop(columns=['item_en']).to_csv('embeddings_gecko_de.csv', index=False)

# retrieve embeddings from gecko model English version
data['embedding'] = embed_text(data['item_en'])
data.rename(columns={'item_en': 'item'}).drop(columns=['item_de']).to_csv('embeddings_gecko_en.csv', index=False)

# retrieve embeddings from latest gecko model (this may not be available and generate an error)
data['embedding'] = embed_text(data['item_de'], 'text-multilingual-embedding-preview-0409')
data.rename(columns={'item_de': 'item'}).drop(columns=['item_en']).to_csv('embeddings_gecko_0409_de.csv', index=False)

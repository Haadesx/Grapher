
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer

_model = None

def get_embeddings(texts: List[str], model_name="all-MiniLM-L6-v2", batch_size=32) -> List[List[float]]:
    """
    Generate embeddings using a local SentenceTransformer model.
    Free and runs locally.
    """
    global _model
    if _model is None:
        print(f"Loading local embedding model: {model_name}...")
        _model = SentenceTransformer(model_name)
        
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = _model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    # Convert to list of lists
    return embeddings.tolist()

def add_embeddings_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'embedding' column to the DataFrame.
    """
    if 'text' not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column")
        
    print(f"Generating embeddings for {len(df)} conversations...")
    embeddings = get_embeddings(df['text'].tolist())
    df['embedding'] = embeddings
    return df

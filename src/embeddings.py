import os
import requests
import time
from typing import List
import pandas as pd
import numpy as np

# Use a standard, small, efficient embedding model hosted on HuggingFace
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"

def get_embeddings_batch(texts: List[str], token: str) -> List[List[float]]:
    """
    Get embeddings from HuggingFace Inference API.
    """
    headers = {"Authorization": f"Bearer {token}"}
    
    # Retry logic
    last_error = ""
    for _ in range(3):
        try:
            response = requests.post(HF_API_URL, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
            if response.status_code == 200:
                return response.json()
            else:
                last_error = f"API Error: {response.status_code} - {response.text}"
                print(last_error)
                time.sleep(2)
        except Exception as e:
            last_error = f"Request failed: {e}"
            print(last_error)
            time.sleep(2)
            
    raise Exception(f"HuggingFace API failed: {last_error}")

def add_embeddings_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an 'embedding' column to the DataFrame using HuggingFace API.
    """
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace Token (HF_TOKEN) is missing. Please add it to your Environment Variables.")

    print(f"Generating embeddings using HuggingFace API ({MODEL_ID})...")
    
    embeddings = []
    batch_size = 20 # Small batch size for API
    texts = df['text'].tolist()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # Truncate text to avoid API limits (approx 512 tokens)
        batch_texts = [t[:1000] for t in batch_texts] 
        
        batch_embeddings = get_embeddings_batch(batch_texts, token)
        
        # Handle API errors or malformed responses
        if isinstance(batch_embeddings, list) and len(batch_embeddings) == len(batch_texts) and isinstance(batch_embeddings[0], list):
             embeddings.extend(batch_embeddings)
        else:
            # If batch fails, fill with empty or zeros? 
            # Better to skip or fill with zeros to keep alignment
            print(f"Warning: Batch {i//batch_size} failed or returned invalid format.")
            embeddings.extend([[] for _ in batch_texts])
            
        print(f"Processed {min(i+batch_size, len(texts))}/{len(texts)}")
        
    df['embedding'] = embeddings
    return df

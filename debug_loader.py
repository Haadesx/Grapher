import json
import os
from src.data_loader import process_conversations

def debug_load():
    input_file = "Export/conversations.json"
    if not os.path.exists(input_file):
        print(f"File {input_file} not found.")
        return

    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"Type of raw_data: {type(raw_data)}")
    if isinstance(raw_data, list):
        print(f"Length of raw_data: {len(raw_data)}")
        if len(raw_data) > 0:
            print(f"Type of first item: {type(raw_data[0])}")
            print(f"Keys of first item: {raw_data[0].keys() if isinstance(raw_data[0], dict) else 'N/A'}")
    elif isinstance(raw_data, dict):
        print(f"Keys of raw_data: {raw_data.keys()}")

    print("Processing conversations...")
    df = process_conversations(raw_data)
    
    print(f"Resulting DataFrame shape: {df.shape}")
    if df.empty:
        print("DataFrame is empty!")
    else:
        print("First few rows:")
        print(df.head())

if __name__ == "__main__":
    debug_load()

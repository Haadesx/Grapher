from memory_profiler import profile
import os
import sys
import psutil
from src.data_loader import load_conversations, process_conversations
from src.embeddings import add_embeddings_to_df
from src.graph_builder import build_similarity_graph

# Set env vars to match main execution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

@profile
def run_pipeline():
    print(f"Initial Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")
    
    input_file = "Export/conversations.json"
    if not os.path.exists(input_file):
        print("File not found!")
        return

    print("1. Loading Data...")
    raw_data = load_conversations(input_file)
    print(f"   Loaded {len(raw_data)} conversations")
    
    print("2. Processing Data...")
    df = process_conversations(raw_data)
    print(f"   Processed {len(df)} items")
    
    print("3. Generating Embeddings...")
    # This is likely the memory peak
    df = add_embeddings_to_df(df)
    
    print("4. Building Graph...")
    G = build_similarity_graph(df)
    
    print(f"Final Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    run_pipeline()

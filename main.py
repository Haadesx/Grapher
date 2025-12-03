import os
import sys
import warnings

# Suppress warnings and logs (Must be before imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress oneDNN custom ops message
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

# Filter specific warnings
warnings.filterwarnings("ignore", message=".*Could not find the number of physical cores.*")
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

from dotenv import load_dotenv
from src.data_loader import load_conversations, process_conversations
from src.embeddings import add_embeddings_to_df
from src.graph_builder import build_similarity_graph
from src.visualizer import generate_visualization

def main():
    load_dotenv()
    
    # Configuration
    input_file = "Export/conversations.json"
    output_file = "index.html"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please place your ChatGPT export file in this directory.")
        return

    # api_key = os.getenv("OPENAI_API_KEY")
    # if not api_key:
    #     print("Warning: OPENAI_API_KEY not found in environment variables.")
    #     # We might want to prompt for it or exit, but for now let's warn.
    #     # The embeddings module will fail if it's not set.
    
    print("Using local embeddings (free).")
    
    print("1. Loading conversations...")
    raw_data = load_conversations(input_file)
    if not raw_data:
        return
        
    print("2. Processing data...")
    df = process_conversations(raw_data)
    print(f"   Found {len(df)} conversations with text.")
    
    print("3. Generating embeddings...")
    try:
        df = add_embeddings_to_df(df)
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return
        
    print("4. Building graph...")
    G = build_similarity_graph(df, threshold=0.3, top_k=5)
    print(f"   Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    print("5. Visualizing...")
    generate_visualization(G, output_file)
    print("Done!")

if __name__ == "__main__":
    main()

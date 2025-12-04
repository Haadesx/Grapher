from flask import Flask, request, jsonify, render_template
import os
import json
import pandas as pd
from src.data_loader import process_conversations
from src.embeddings import add_embeddings_to_df
from src.graph_builder import build_similarity_graph
import networkx as nx

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file:
        try:
            # Load JSON directly from stream
            raw_data = json.load(file)
            
            # Process Data
            df = process_conversations(raw_data)
            if df.empty:
                # Generate a helpful error message
                debug_msg = "No valid conversations found."
                if isinstance(raw_data, list):
                    debug_msg += f" Uploaded list with {len(raw_data)} items."
                    if len(raw_data) > 0:
                        first_item = raw_data[0]
                        if isinstance(first_item, dict):
                            debug_msg += f" First item keys: {list(first_item.keys())}."
                        else:
                            debug_msg += f" First item type: {type(first_item)}."
                elif isinstance(raw_data, dict):
                    debug_msg += f" Uploaded dict with keys: {list(raw_data.keys())}."
                else:
                    debug_msg += f" Uploaded data type: {type(raw_data)}."
                    
                return jsonify({'error': debug_msg}), 400
                
            # Generate Embeddings (This might be slow)
            df = add_embeddings_to_df(df)
            
            # Build Graph
            G = build_similarity_graph(df, threshold=0.3, top_k=5)
            
            # Convert to JSON
            graph_data = nx.node_link_data(G)
            
            return jsonify(graph_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

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
                return jsonify({'error': 'No valid conversations found'}), 400
                
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

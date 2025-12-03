# ChatGPT 3D Graph Visualization

A Python tool that visualizes your ChatGPT conversation history as an interactive 3D galaxy.

## Features
- **3D Force-Directed Graph:** Visualizes conversations as nodes in a 3D space.
- **Semantic Clustering:** Groups similar conversations together using K-Means clustering on embeddings.
- **Local Embeddings:** Uses `sentence-transformers` (all-MiniLM-L6-v2) for free, local embedding generation.
- **Interactive UI:** Click on nodes to see the conversation title, date, message count, and a snippet of the first query.
- **Cyberpunk Aesthetic:** Dark mode, glowing particles, and smooth camera controls.

## Usage
1.  Export your data from ChatGPT (Settings -> Data Controls -> Export Data).
2.  Place the `conversations.json` file in the `Export/` directory.
3.  Run the script:
    ```bash
    pip install -r requirements.txt
    python main.py
    ```
4.  Open `index.html` in your browser.

## Tech Stack
- **Python:** Pandas, NumPy, NetworkX, Scikit-learn, Torch
- **Frontend:** 3d-force-graph.js, Three.js, HTML/CSS

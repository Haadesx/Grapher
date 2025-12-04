import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

def extract_cluster_keywords(df, n_clusters=8, top_n=3):
    """
    Extracts top keywords for each cluster using TF-IDF-like logic (or simple frequency).
    """
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['text'])
    feature_names = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    for cluster_id in range(n_clusters):
        # Get all text for this cluster
        cluster_indices = df[df['cluster'] == cluster_id].index
        if len(cluster_indices) == 0:
            cluster_keywords[cluster_id] = "Misc"
            continue
            
        # Sum word counts for this cluster
        cluster_word_counts = X[cluster_indices].sum(axis=0)
        
        # Get top N words
        top_indices = np.argsort(np.asarray(cluster_word_counts).flatten())[::-1][:top_n]
        keywords = [feature_names[i] for i in top_indices]
        cluster_keywords[cluster_id] = ", ".join(keywords).title()
        
    return cluster_keywords

def build_similarity_graph(df: pd.DataFrame, threshold=0.3, top_k=5) -> nx.Graph:
    """
    Builds a NetworkX graph where nodes are conversations and edges represent similarity.
    Enriches nodes with cluster info and keywords.
    """
    G = nx.Graph()
    
    # Filter out rows with empty embeddings
    valid_df = df[df['embedding'].apply(lambda x: len(x) > 0)].copy()
    
    # 1. Clustering & Node Addition
    if len(valid_df) < 2:
        # Single node case - no clustering or edges needed
        row = valid_df.iloc[0]
        G.add_node(
            row['id'], 
            title=row['title'], 
            group=0,
            cluster_label="Single Conversation",
            snippet=row['snippet'],
            message_count=row['message_count'],
            date=pd.to_datetime(row['create_time'], unit='s').strftime('%Y-%m-%d') if pd.notnull(row['create_time']) else "Unknown"
        )
        return G

    print("   Running K-Means clustering...")
    embeddings_matrix = np.vstack(valid_df['embedding'].values)
    n_clusters = max(1, min(8, len(valid_df) // 5)) # Ensure at least 1 cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    valid_df['cluster'] = kmeans.fit_predict(embeddings_matrix)
    
    # 2. Keyword Extraction
    print("   Extracting cluster topics...")
    cluster_labels = extract_cluster_keywords(valid_df, n_clusters=n_clusters)
    
    # Add nodes with rich metadata
    for idx, row in valid_df.iterrows():
        cluster_name = cluster_labels.get(row['cluster'], f"Group {row['cluster']}")
        
        G.add_node(
            row['id'], 
            title=row['title'], 
            group=int(row['cluster']),
            cluster_label=cluster_name,
            snippet=row['snippet'], # Use the pre-extracted snippet
            message_count=row['message_count'],
            date=pd.to_datetime(row['create_time'], unit='s').strftime('%Y-%m-%d') if pd.notnull(row['create_time']) else "Unknown"
        )
        
    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings_matrix)
    
    # Add edges
    ids = valid_df['id'].tolist()
    
    for i in range(len(ids)):
        # Get indices of top_k similar items (excluding self)
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        
        count = 0
        for j in sorted_indices:
            if i == j:
                continue
                
            sim_score = sim_matrix[i][j]
            
            if sim_score < threshold:
                break 
                
            if count >= top_k:
                break
                
            # Add edge
            G.add_edge(ids[i], ids[j], weight=float(sim_score), value=float(sim_score))
            count += 1
            
    return G

import json
import networkx as nx
from jinja2 import Template

HTML_TEMPLATE = """
<head>
  <style> 
    body { margin: 0; background-color: #000005; overflow: hidden; font-family: 'Segoe UI', sans-serif; } 
    #3d-graph { width: 100vw; height: 100vh; }
    
    /* UI Overlay */
    #info-panel {
        position: absolute;
        top: 20px;
        right: 20px;
        width: 320px;
        background: rgba(0, 5, 16, 0.9);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 255, 204, 0.2);
        border-radius: 16px;
        padding: 24px;
        color: #e0e0e0;
        transform: translateX(120%);
        transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 0 40px rgba(0, 255, 204, 0.1);
        z-index: 10;
        font-family: 'Segoe UI', sans-serif;
    }
    
    #info-panel.visible {
        transform: translateX(0);
    }

    .meta-header {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #888;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    h2 {
        margin: 0 0 8px 0;
        color: #fff;
        font-size: 20px;
        font-weight: 600;
        line-height: 1.3;
        text-shadow: 0 0 15px rgba(0, 255, 204, 0.3);
    }
    
    .cluster-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 20px;
        border: 1px solid;
        background: rgba(255,255,255,0.05);
    }
    
    .content-box {
        border-left: 3px solid #00ffcc;
        padding-left: 15px;
        margin-top: 10px;
        background: rgba(255, 255, 255, 0.02);
        padding: 15px;
        border-radius: 0 8px 8px 0;
    }

    p {
        font-size: 14px;
        line-height: 1.6;
        color: #ccc;
        margin: 0;
        font-style: italic;
    }
    
    /* Instructions */
    #instructions {
        position: absolute;
        bottom: 20px;
        left: 20px;
        color: rgba(255, 255, 255, 0.4);
        font-size: 12px;
        pointer-events: none;
    }
  </style>
  
  <script src="https://unpkg.com/3d-force-graph"></script>
  <script src="https://unpkg.com/three"></script>
  <script src="https://unpkg.com/three-spritetext"></script>
</head>

<body>
  <div id="3d-graph"></div>
  
  <div id="instructions">
    Left Click: Rotate | Right Click: Pan | Scroll: Zoom | Click Node: Details
  </div>

  <div id="info-panel">
    <div class="meta-header">
        <span id="panel-date">Date</span>
        <span id="panel-msgs">0 Msgs</span>
    </div>
    <h2 id="panel-title">Select a Node</h2>
    <div id="panel-cluster" class="cluster-tag">Topic</div>
    
    <div class="content-box">
        <p id="panel-content">Click on any node in the galaxy to explore its contents.</p>
    </div>
  </div>

  <script>
    const gData = {{ graph_data | safe }};

    // Color Palette (Cyberpunk/Space)
    const colors = [
        '#00ffcc', // Cyan
        '#bf00ff', // Purple
        '#ff0055', // Pink
        '#ffcc00', // Gold
        '#00ccff', // Blue
        '#ff6600', // Orange
        '#00ff66', // Green
        '#ff00cc'  // Magenta
    ];

    const Graph = ForceGraph3D()
      (document.getElementById('3d-graph'))
        .graphData(gData)
        .backgroundColor('#000005')
        
        // Nodes
        .nodeLabel('title')
        .nodeRelSize(6)
        .nodeColor(node => colors[node.group % colors.length])
        .nodeOpacity(0.9)
        .nodeResolution(32)
        
        // Links
        .linkWidth(0.6)
        .linkOpacity(0.15)
        .linkColor(() => '#ffffff')
        
        // Particles
        .linkDirectionalParticles(2)
        .linkDirectionalParticleWidth(2)
        .linkDirectionalParticleSpeed(0.005)
        .linkDirectionalParticleColor(link => colors[link.source.group % colors.length])
        
        // Interaction
        .onNodeClick(node => {
            // Update Panel
            const panel = document.getElementById('info-panel');
            const color = colors[node.group % colors.length]; // Define color variable
            
            document.getElementById('panel-title').innerText = node.title;
            
            const clusterTag = document.getElementById('panel-cluster');
            clusterTag.innerText = node.cluster_label || 'Unknown Topic';
            clusterTag.style.color = color;
            clusterTag.style.borderColor = color;
            clusterTag.style.background = color + '22'; // Low opacity background
            
            // Format snippet for glanceability
            let snippet = node.snippet || 'No preview available.';
            // Remove "User: " prefix if present for cleaner look
            snippet = snippet.replace(/^User:\s*/i, '');
            document.getElementById('panel-content').innerText = snippet;
            
            document.getElementById('panel-date').innerText = node.date || 'Unknown';
            document.getElementById('panel-msgs').innerText = (node.message_count || 0) + ' Msgs';
            
            document.querySelector('.content-box').style.borderLeftColor = color;
            
            panel.classList.add('visible');

            // Camera Fly
            const distance = 60;
            const distRatio = 1 + distance/Math.hypot(node.x, node.y, node.z);

            Graph.cameraPosition(
                { x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio },
                node, 
                2000
            );
        })
        .onBackgroundClick(() => {
            document.getElementById('info-panel').classList.remove('visible');
        });

    // Add Stars
    // We can access the ThreeJS scene via Graph.scene()
    // But 3d-force-graph doesn't expose it easily in the builder pattern without a custom object.
    // However, the default black void with particles is already quite "space-like".
    
    // Auto-rotate
    Graph.controls().autoRotate = true;
    Graph.controls().autoRotateSpeed = 0.6;
    
  </script>
</body>
"""

def generate_visualization(G: nx.Graph, output_file="graph.html"):
    """
    Generates a standalone HTML file with the 3D graph visualization.
    """
    # Convert NetworkX graph to node-link data format
    data = nx.node_link_data(G)
    
    # Render template
    template = Template(HTML_TEMPLATE)
    html_content = template.render(graph_data=json.dumps(data))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Visualization saved to {output_file}")

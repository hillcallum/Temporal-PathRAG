"""
Temporal PathRAG Demo
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
import sys
from typing import List, Dict, Set
import random

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import actual Temporal PathRAG components
from src.kg.storage.updated_tkg_loader import load_enhanced_dataset

st.set_page_config(page_title="Temporal PathRAG Demo", layout="wide", initial_sidebar_state="collapsed")

# Fix styling 
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: none;
    }
    h1 { font-size: 24px; margin-bottom: 10px; }
    .stMarkdown { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_knowledge_graph():
    """Load the knowledge graph"""
    graph = load_enhanced_dataset("MultiTQ")
    return graph

def get_initial_subgraph(graph: nx.DiGraph, centre_node: str = "Abdul_Kalam", max_nodes: int = 100) -> nx.DiGraph:
    """Get an initial subgraph centred around a node"""
    if centre_node not in graph:
        # If centre node not found, use a random sample
        nodes = list(graph.nodes())[:max_nodes]
        return graph.subgraph(nodes)
    
    # BFS to get nearby nodes
    nodes_to_include = set([centre_node])
    current_layer = set([centre_node])
    
    while len(nodes_to_include) < max_nodes and current_layer:
        next_layer = set()
        for node in current_layer:
            if node in graph:
                # Add neighbours
                neighbours = list(graph.neighbors(node))[:5]
                next_layer.update(neighbours)
                # Add predecessors
                predecessors = list(graph.predecessors(node))[:5]
                next_layer.update(predecessors)
        
        # Add nodes from next layer up to max_nodes
        for node in next_layer:
            if len(nodes_to_include) >= max_nodes:
                break
            nodes_to_include.add(node)
        
        current_layer = next_layer
    
    return graph.subgraph(list(nodes_to_include))

def get_demo_query():
    """Get a demo query with iterative reasoning steps"""
    return {
        "query": "Which government officials did Abdul Kalam work with in India?",
        "iterations": [
            {
                "step": 1,
                "sub_question": "Who is Abdul Kalam?",
                "reasoning": "First, identify the entity Abdul Kalam in the knowledge graph",
                "entities_found": ["Abdul_Kalam"],
                "pruning_strategy": "Keep only nodes within 2 hops of Abdul_Kalam"
            },
            {
                "step": 2,
                "sub_question": "What organizations did Abdul Kalam work for?",
                "reasoning": "Find employment/affiliation relationships",
                "entities_found": ["Government_of_India", "India"],
                "pruning_strategy": "Filter to government-related organizations"
            },
            {
                "step": 3,
                "sub_question": "Who were the officials in these organizations?",
                "reasoning": "Find people connected to these government organizations",
                "entities_found": ["Prime_Minister_(India)", "Head_of_Government_(India)"],
                "pruning_strategy": "Keep only person entities with official roles"
            },
            {
                "step": 4,
                "sub_question": "Which officials did Abdul Kalam directly interact with?",
                "reasoning": "Find direct paths between Abdul Kalam and officials",
                "entities_found": ["Prime_Minister_(India)", "Head_of_Government_(India)"],
                "pruning_strategy": "Keep only paths with direct interactions"
            }
        ],
        "final_answer": "Abdul Kalam worked with the Prime Minister and Head of Government of India"
    }

def create_graph_visualisation(graph: nx.DiGraph, 
                             highlight_nodes: Set[str] = None,
                             pruned_nodes: Set[str] = None,
                             active_paths: List[List[str]] = None,
                             iteration: int = 0) -> go.Figure:
    """Create a clean, interactive graph visualisation"""
    
    # Determine which nodes to show
    if pruned_nodes:
        visible_nodes = [n for n in graph.nodes() if n not in pruned_nodes]
    else:
        visible_nodes = list(graph.nodes())
    
    # Create subgraph
    subgraph = graph.subgraph(visible_nodes)
    
    # Create layout
    if len(subgraph) == 0:
        # Empty graph
        fig = go.Figure()
        fig.add_annotation(
            text="No nodes to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20, color='gray')
        )
        fig.update_layout(
            plot_bgcolor='white',
            height=700,
            margin=dict(t=0, r=0, b=0, l=0)
        )
        return fig
    
    # Use spring layout for positioning
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
    
    # Extract active path edges
    active_edges = set()
    if active_paths:
        for path in active_paths:
            for i in range(len(path) - 1):
                if path[i] in subgraph and path[i+1] in subgraph:
                    active_edges.add((path[i], path[i+1]))
    
    fig = go.Figure()
    
    # Draw edges
    edge_x = []
    edge_y = []
    path_edge_x = []
    path_edge_y = []
    
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        if edge in active_edges or (edge[1], edge[0]) in active_edges:
            path_edge_x.extend([x0, x1, None])
            path_edge_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    # Add normal edges
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#e0e0e0'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Add path edges
    if path_edge_x:
        fig.add_trace(go.Scatter(
            x=path_edge_x, y=path_edge_y,
            mode='lines',
            line=dict(width=3, color='#e74c3c'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Prepare node data
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    hover_text = []
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node styling
        if highlight_nodes and node in highlight_nodes:
            if node == "Abdul_Kalam":
                node_color.append('#2ecc71')  # Green for source
                node_size.append(25)
            elif iteration == 4 and "Prime_Minister" in node or "Head_of_Government" in node:
                node_color.append('#e74c3c')  # Red for answers
                node_size.append(20)
            else:
                node_color.append('#3498db')  # Blue for highlighted
                node_size.append(15)
        else:
            node_color.append('#bdc3c7')  # Light gray for others
            node_size.append(8)
        
        # Node label
        label = str(node).replace('_', ' ')
        if len(label) > 30:
            label = label[:27] + '...'
        
        # Only show labels for highlighted nodes
        if highlight_nodes and node in highlight_nodes:
            node_text.append(label)
        else:
            node_text.append('')
        
        # Hover information
        hover = f"<b>{node}</b><br>"
        hover += f"Connections: {subgraph.degree(node)}"
        hover_text.append(hover)
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top centre",
        textfont=dict(size=10, color='#2c3e50'),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color='white')
        ),
        hovertext=hover_text,
        hoverinfo='text',
        hoverlabel=dict(bgcolor='white', font_size=12),
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(t=20, r=20, b=20, l=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=650,
        dragmode='pan'
    )
    
    # Add graph stats
    fig.add_annotation(
        text=f"Nodes: {len(visible_nodes)} | Edges: {len(subgraph.edges())}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12, color='#7f8c8d'),
        bgcolor='rgba(255,255,255,0.8)',
        borderpad=4
    )
    
    return fig

def apply_pruning_strategy(graph: nx.DiGraph, full_graph: nx.DiGraph, iteration: Dict) -> Set[str]:
    """Apply pruning strategy for current iteration"""
    step = iteration['step']
    
    if step == 1:
        # Keep only nodes within 2 hops of Abdul_Kalam
        keep_nodes = set(['Abdul_Kalam'])
        current_layer = set(['Abdul_Kalam'])
        
        for _ in range(2):
            next_layer = set()
            for node in current_layer:
                if node in graph:
                    next_layer.update(list(graph.neighbors(node))[:10])
                    next_layer.update(list(graph.predecessors(node))[:10])
            keep_nodes.update(next_layer)
            current_layer = next_layer
        
        return set(graph.nodes()) - keep_nodes
        
    elif step == 2:
        # Keep government organisations and related nodes
        keep_nodes = set()
        for node in graph.nodes():
            node_str = str(node).lower()
            if any(term in node_str for term in ['government', 'india', 'abdul']):
                keep_nodes.add(node)
                # Add neighbours
                if node in graph:
                    keep_nodes.update(list(graph.neighbors(node))[:5])
        
        return set(graph.nodes()) - keep_nodes
        
    elif step == 3:
        # Keep only person entities and officials
        keep_nodes = set(['Abdul_Kalam'])
        for node in graph.nodes():
            node_str = str(node).lower()
            if any(term in node_str for term in ['minister', 'prime', 'president', 'head_of_government', 'india']):
                keep_nodes.add(node)
        
        # Add connections between these nodes
        for node in list(keep_nodes):
            if node in graph:
                for neighbour in graph.neighbors(node):
                    if any(term in str(neighbour).lower() for term in ['minister', 'government', 'india']):
                        keep_nodes.add(neighbour)
        
        return set(graph.nodes()) - keep_nodes
        
    elif step == 4:
        # Keep only direct paths
        keep_nodes = {'Abdul_Kalam'}
        target_nodes = set()
        
        for node in graph.nodes():
            if 'Prime_Minister' in node or 'Head_of_Government' in node:
                target_nodes.add(node)
        
        # Find short paths
        for target in target_nodes:
            if target in graph:
                try:
                    for path in nx.all_simple_paths(graph, 'Abdul_Kalam', target, cutoff=3):
                        keep_nodes.update(path)
                except:
                    pass
        
        return set(graph.nodes()) - keep_nodes
    
    return set()

def main():
    st.title("Temporal PathRAG: Iterative Reasoning Demo")
    
    # Load full graph
    with st.spinner("Loading knowledge graph"):
        full_graph = load_knowledge_graph()
    
    # Get initial subgraph for visualisation
    initial_graph = get_initial_subgraph(full_graph, "Abdul_Kalam", max_nodes=100)
    
    # Get demo query
    query_data = get_demo_query()
    
    # Layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(f"**Query:** {query_data['query']}")
        st.markdown("---")
        
        # Iteration control
        iteration = st.slider("Reasoning Step", 0, 4, 0, key="iteration_slider")
        
        if iteration == 0:
            st.markdown("**Initial State**")
            st.markdown("Subgraph centred on Abdul Kalam")
            st.markdown(f"- Showing: {len(initial_graph.nodes())} nodes")
            st.markdown(f"- Total graph: {len(full_graph.nodes())} nodes")
        else:
            current_iteration = query_data['iterations'][iteration - 1]
            
            st.markdown(f"**Step {iteration}:** {current_iteration['sub_question']}")
            st.markdown(f"*{current_iteration['reasoning']}*")
            
            st.markdown("**Entities Found:**")
            for entity in current_iteration['entities_found']:
                st.markdown(f"- `{entity}`")
            
            st.markdown("**Pruning:**")
            st.markdown(current_iteration['pruning_strategy'])
            
            if iteration == 4:
                st.markdown("---")
                st.markdown(f"**Answer:** {query_data['final_answer']}")
    
    with col2:
        # Create visualisation
        if iteration == 0:
            # Show initial subgraph
            fig = create_graph_visualisation(initial_graph, iteration=iteration)
        else:
            # Apply iterative pruning
            current_iter = query_data['iterations'][iteration - 1]
            highlight_nodes = set(current_iter['entities_found'])
            highlight_nodes.add('Abdul_Kalam')
            
            # Apply pruning to the subgraph
            pruned_nodes = apply_pruning_strategy(initial_graph, full_graph, current_iter)
            
            # Create paths for final iteration
            active_paths = []
            if iteration == 4:
                active_paths = [
                    ['Abdul_Kalam', 'India', 'Prime_Minister_(India)'],
                    ['Abdul_Kalam', 'Government_of_India', 'Head_of_Government_(India)']
                ]
            
            fig = create_graph_visualisation(
                initial_graph,
                highlight_nodes=highlight_nodes,
                pruned_nodes=pruned_nodes,
                active_paths=active_paths,
                iteration=iteration
            )
        
        # Display graph
        st.plotly_chart(fig, use_container_width=True)
        
        # Legend
        if iteration > 0:
            st.markdown(
                "Green = Source ; Blue = Found Entities ; Red = Answer ; White = Context",
                help="Node colors indicate their role in the reasoning process"
            )

if __name__ == "__main__":
    main()
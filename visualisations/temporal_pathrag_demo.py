"""
Temporal PathRAG Demo
Visualisation using actual complex queries from evaluation dataset
"""

import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import json
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Tuple
import random

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

st.set_page_config(page_title="Temporal PathRAG Demo", layout="wide")

# Clean styling
st.markdown("""
<style>
    .stApp { margin: 0; padding: 1rem; }
    h1 { font-size: 2rem; margin-bottom: 1rem; }
    .main > div { padding: 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_knowledge_graph():
    """Load the knowledge graph"""
    try:
        from src.kg.storage.updated_tkg_loader import load_enhanced_dataset
        graph = load_enhanced_dataset("MultiTQ")
        return graph, True
    except Exception as e:
        st.warning(f"Could not load real graph: {e}. Using synthetic data.")
        return create_synthetic_graph(), False

@st.cache_data
def load_complex_queries():
    """Load complex queries from evaluation dataset"""
    query_file = Path("evaluation/complex_temporal_queries.json")
    if query_file.exists():
        with open(query_file, 'r') as f:
            queries = json.load(f)
            # Return queries with reasoning steps
            return [q for q in queries if 'metadata' in q and 'reasoning_steps' in q['metadata']]
    else:
        st.error("Complex queries file not found!")
        return []

def create_synthetic_graph():
    """Fallback synthetic graph"""
    G = nx.DiGraph()
    
    # Add some basic nodes for demo
    people = ['Joe Biden', 'Emmanuel Macron', 'Olaf Scholz', 'Volodymyr Zelensky']
    events = ['Ukraine War', 'NATO Summit', 'G7 Meeting']
    
    for person in people:
        G.add_node(person, type='Person')
    for event in events:
        G.add_node(event, type='Event')
    
    # Add edges
    G.add_edge('Joe Biden', 'NATO Summit', date='2022-03-24')
    G.add_edge('Emmanuel Macron', 'NATO Summit', date='2022-03-24')
    G.add_edge('Ukraine War', 'Volodymyr Zelensky', date='2022-02-24')
    
    return G

def extract_entities_from_query(query: str, reasoning_steps: List[str]) -> Dict[str, Any]:
    """Extract entities and constraints from query and reasoning steps"""
    entities = []
    temporal_constraints = []
    
    # Simple extraction based on common patterns
    query_lower = query.lower()
    
    # Extract person names (basic heuristic)
    if "joe biden" in query_lower:
        entities.append("Joe Biden")
    if "emmanuel macron" in query_lower:
        entities.append("Emmanuel Macron")
    if "european leaders" in query_lower:
        entities.extend(["Emmanuel Macron", "Olaf Scholz", "Boris Johnson"])
    
    # Extract events
    if "ukraine war" in query_lower:
        entities.append("Ukraine War")
        temporal_constraints.append("After February 24, 2022")
    if "crimea" in query_lower:
        temporal_constraints.append("After March 2014")
    
    # Extract from reasoning steps
    for step in reasoning_steps:
        if "Feb" in step or "February" in step:
            if "2022" in step:
                temporal_constraints.append("After February 24, 2022")
        if "March" in step and "2014" in step:
            temporal_constraints.append("After March 2014")
    
    return {
        'entities': entities,
        'temporal_constraints': temporal_constraints
    }

def visualise_graph_interactive(graph: nx.DiGraph, 
                              highlight_nodes: List[str] = [],
                              subset_nodes: Optional[List[str]] = None,
                              title: str = "",
                              max_nodes: int = 500):
    """Create an interactive graph visualisation"""
    
    # Handle subset
    if subset_nodes:
        subgraph = graph.subgraph(subset_nodes)
    else:
        # Sample if too large
        if len(graph.nodes()) > max_nodes:
            all_nodes = list(graph.nodes())
            sampled = set(highlight_nodes) if highlight_nodes else set()
            
            remaining = [n for n in all_nodes if n not in sampled]
            sample_size = min(max_nodes - len(sampled), len(remaining))
            if sample_size > 0:
                sampled.update(random.sample(remaining, sample_size))
            
            subgraph = graph.subgraph(list(sampled))
        else:
            subgraph = graph
    
    # Layout
    pos = nx.spring_layout(subgraph, k=3 if len(subgraph.nodes()) > 50 else 2, 
                          iterations=30, seed=42)
    
    fig = go.Figure()
    
    # Draw edges
    for u, v, data in subgraph.edges(data=True):
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))
        
        edge_date = data.get('date', '')
        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=0.5, colour='rgba(200,200,200,0.5)'),
            hoverinfo='text',
            hovertext=f"{u} → {v}<br>Date: {edge_date}" if edge_date else f"{u} → {v}",
            showlegend=False
        )
        fig.add_trace(edge_trace)
    
    # Prepare node data
    node_x = []
    node_y = []
    node_colour = []
    node_size = []
    hover_text = []
    
    for node in subgraph.nodes():
        x, y = pos.get(node, (0, 0))
        node_x.append(x)
        node_y.append(y)
        
        # Node styling
        if node in highlight_nodes:
            node_colour.append('#ff4444')
            node_size.append(15)
        else:
            node_type = subgraph.nodes[node].get('type', 'Unknown')
            if node_type == 'Person':
                node_colour.append('#4A90E2')
            elif node_type == 'Event':
                node_colour.append('#50C878')
            else:
                node_colour.append('#95A5A6')
            node_size.append(8)
        
        # Hover text
        node_data = subgraph.nodes[node]
        text = f"<b>{node}</b><br>"
        text += f"Type: {node_data.get('type', 'Unknown')}<br>"
        text += f"Connections: {subgraph.degree(node)}"
        hover_text.append(text)
    
    # Draw nodes
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_size,
            colour=node_colour,
            line=dict(width=1, colour='white')
        ),
        text=[str(n)[:20] + '...' if len(str(n)) > 20 else str(n) for n in subgraph.nodes()],
        textposition='top center',
        textfont=dict(size=8),
        hovertext=hover_text,
        hoverinfo='text',
        hoverlabel=dict(bgcolour='white', font_size=12),
        showlegend=False
    )
    fig.add_trace(node_trace)
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        showlegend=False,
        hovermode='closest',
        margin=dict(t=30, r=0, b=0, l=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolour='white',
        height=600,
        dragmode='pan'
    )
    
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['toImage', 'autoScale2d']
    }
    
    return fig, config

def show_reasoning_step(graph: nx.DiGraph, step_num: int, query_data: Dict[str, Any]):
    """Show a specific reasoning step"""
    query = query_data['query']
    reasoning_steps = query_data['metadata']['reasoning_steps']
    
    # Extract entities and constraints
    extracted = extract_entities_from_query(query, reasoning_steps)
    entities = extracted['entities']
    
    # Ensure entities exist in graph
    existing_entities = [e for e in entities if e in graph]
    if not existing_entities and len(graph.nodes()) > 0:
        # Use random entities as fallback
        existing_entities = random.sample(list(graph.nodes()), min(3, len(graph.nodes())))
    
    if step_num == 0:
        # Full graph
        fig, config = visualise_graph_interactive(
            graph,
            title=f"Complete Knowledge Graph ({len(graph.nodes()):,} nodes, {len(graph.edges()):,} edges)",
            max_nodes=500
        )
    
    elif step_num == 1:
        # Extract entities
        subset = set(existing_entities)
        for entity in existing_entities:
            if entity in graph:
                neighbours = list(graph.neighbors(entity))[:10]
                subset.update(neighbours)
        
        subset_list = list(subset)[:100]
        fig, config = visualise_graph_interactive(
            graph,
            highlight_nodes=existing_entities,
            subset_nodes=subset_list,
            title=f"Step 2: {reasoning_steps[1] if len(reasoning_steps) > 1 else 'Extract Key Entities'} ({len(subset_list)} nodes)"
        )
    
    elif step_num == 2:
        # Temporal filter
        subset = set(existing_entities[:2])
        for entity in existing_entities[:2]:
            if entity in graph:
                neighbours = list(graph.neighbors(entity))[:5]
                subset.update(neighbours)
        
        subset_list = list(subset)[:50]
        fig, config = visualise_graph_interactive(
            graph,
            highlight_nodes=existing_entities,
            subset_nodes=subset_list,
            title=f"Step 3: {reasoning_steps[2] if len(reasoning_steps) > 2 else 'Apply Temporal Filter'} ({len(subset_list)} nodes)"
        )
    
    elif step_num == 3:
        # Path finding
        path_nodes = existing_entities[:3] if len(existing_entities) >= 3 else existing_entities
        subset = path_nodes + random.sample(list(set(graph.nodes()) - set(path_nodes)), 
                                          min(3, len(set(graph.nodes()) - set(path_nodes))))
        
        fig, config = visualise_graph_interactive(
            graph,
            highlight_nodes=path_nodes,
            subset_nodes=subset[:10],
            title=f"Step 4: {reasoning_steps[3] if len(reasoning_steps) > 3 else 'Find Relevant Paths'} ({len(subset)} nodes)"
        )
    
    else:
        # Final answer
        answer = query_data.get('answer', 'Emmanuel Macron and Olaf Scholz')
        # Extract answer entities
        answer_entities = []
        for entity in existing_entities:
            if entity in answer:
                answer_entities.append(entity)
        
        if not answer_entities:
            answer_entities = existing_entities[:2]
        
        fig, config = visualise_graph_interactive(
            graph,
            highlight_nodes=answer_entities,
            subset_nodes=answer_entities,
            title=f"Final Answer: {answer}"
        )
    
    return fig, config

def main():
    # Hide Streamlit components
    hide_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(hide_style, unsafe_allow_html=True)
    
    # Title
    st.markdown("<h1 style='text-align: center;'>Temporal PathRAG Demo</h1>", unsafe_allow_html=True)
    
    # Load data
    graph, is_real = load_knowledge_graph()
    queries = load_complex_queries()
    
    if not queries:
        st.error("No complex queries found! Please ensure evaluation/complex_temporal_queries.json exists.")
        return
    
    # Initialise session state
    if 'query_idx' not in st.session_state:
        st.session_state.query_idx = 0
    if 'step' not in st.session_state:
        st.session_state.step = 0
    
    # Get current query
    current_query = queries[st.session_state.query_idx]
    
    # Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"**Query {st.session_state.query_idx + 1} of {len(queries)}:**")
        st.write(current_query['query'])
        
        st.markdown("**Query Type:** " + current_query.get('type', 'Unknown'))
        st.markdown("**Complexity:** " + current_query.get('complexity', 'Unknown'))
        
        st.markdown("**Reasoning Steps:**")
        reasoning_steps = current_query['metadata']['reasoning_steps']
        
        # Show all steps with current highlighted
        all_steps = ["Full Knowledge Graph"] + reasoning_steps[:4]
        for i, step in enumerate(all_steps):
            if i <= st.session_state.step:
                st.markdown(f"**{i+1}. {step}**")
            else:
                st.markdown(f"<span style='colour: gray;'>{i+1}. {step}</span>", 
                          unsafe_allow_html=True)
        
        # Controls
        st.markdown("<br>", unsafe_allow_html=True)
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("← Previous Step", disabled=st.session_state.step == 0):
                st.session_state.step -= 1
                st.rerun()
        with col_next:
            if st.button("Next Step →", disabled=st.session_state.step >= 4):
                st.session_state.step += 1
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Query navigation
        st.markdown("**Navigate Queries:**")
        col_prev_q, col_next_q = st.columns(2)
        with col_prev_q:
            if st.button("← Previous Query", disabled=st.session_state.query_idx == 0):
                st.session_state.query_idx -= 1
                st.session_state.step = 0
                st.rerun()
        with col_next_q:
            if st.button("Next Query →", disabled=st.session_state.query_idx >= len(queries)-1):
                st.session_state.query_idx += 1
                st.session_state.step = 0
                st.rerun()
    
    with col2:
        # Show graph for current step
        fig, config = show_reasoning_step(graph, st.session_state.step, current_query)
        st.plotly_chart(fig, use_container_width=True, config=config)

if __name__ == "__main__":
    main()
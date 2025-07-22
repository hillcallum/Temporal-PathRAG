#!/usr/bin/env python3
"""
Matplotlib Visualisation Script for Temporal KG
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.kg.storage.temporal_graph_storage import TemporalGraphDatabase

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_temporal_overview_plots(tg_db: TemporalGraphDatabase):
    """Create overview plots showing temporal distribution and statistics"""
    
    print("Creating temporal overview plots")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Temporal Knowledge Graph Overview', fontsize=16, fontweight='bold')
    
    # 1. Timeline of facts over time
    temporal_data = []
    for timestamp, edge_keys in tg_db.temporal_index.items():
        try:
            if '-' in timestamp:
                year = int(timestamp.split('-')[0])
            elif len(timestamp) >= 4:
                year = int(timestamp[:4])
            else:
                continue
            temporal_data.append({
                'year': year,
                'fact_count': len(edge_keys)
            })
        except:
            continue
    
    df_temporal = pd.DataFrame(temporal_data)
    yearly_facts = df_temporal.groupby('year')['fact_count'].sum().reset_index()
    
    # Focus on reasonable time range
    yearly_facts = yearly_facts[(yearly_facts['year'] >= 1900) & (yearly_facts['year'] <= 2025)]
    
    ax1.plot(yearly_facts['year'], yearly_facts['fact_count'], marker='o', linewidth=2, markersize=3)
    ax1.set_title('Facts Over Time (1900-2025)', fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Facts')
    ax1.grid(True, alpha=0.3)
    
    # 2. Top 15 Relations
    rel_stats = tg_db.get_relation_statistics()
    top_relations = list(rel_stats['relation_frequency'].items())[:15]
    
    relations = [r[0][:20] + '...' if len(r[0]) > 20 else r[0] for r in top_relations]
    counts = [r[1] for r in top_relations]
    
    bars = ax2.barh(range(len(relations)), counts)
    ax2.set_yticks(range(len(relations)))
    ax2.set_yticklabels(relations, fontsize=8)
    ax2.set_title('Top 15 Relations by Frequency', fontweight='bold')
    ax2.set_xlabel('Frequency')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + max(counts)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{counts[i]:,}', ha='left', va='center', fontsize=7)
    
    # 3. Dataset Distribution
    dataset_counts = {}
    for u, v, data in tg_db.main_graph.edges(data=True):
        dataset = data.get('dataset', 'unknown')
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    datasets = list(dataset_counts.keys())
    values = list(dataset_counts.values())
    colours = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    wedges, texts, autotexts = ax3.pie(values, labels=datasets, autopct='%1.1f%%', 
                                      colours=colours[:len(datasets)], startangle=90)
    ax3.set_title('Edge Distribution by Dataset', fontweight='bold')
    
    # 4. Entity Degree Distribution
    node_degrees = [tg_db.main_graph.degree(node) for node in tg_db.main_graph.nodes()]
    
    ax4.hist(node_degrees, bins=50, alpha=0.7, colour='skyblue', edgecolour='black')
    ax4.set_title('Entity Degree Distribution', fontweight='bold')
    ax4.set_xlabel('Node Degree')
    ax4.set_ylabel('Frequency')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_degree = np.mean(node_degrees)
    max_degree = max(node_degrees)
    ax4.text(0.7, 0.8, f'Mean: {mean_degree:.1f}\nMax: {max_degree}', 
            transform=ax4.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round', facecolour='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    Path("analysis_results/plots").mkdir(exist_ok=True)
    output_file = "analysis_results/plots/temporal_kg_overview.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Overview visualisation saved to: {output_file}")
    plt.close()
    
    return output_file

def create_entity_subgraph_plot(tg_db: TemporalGraphDatabase, entity: str, max_nodes: int = 25):
    """Create a subgraph visualisation around a specific entity"""
    
    print(f"Creating subgraph plot for entity: {entity}")
    
    if not tg_db.main_graph.has_node(entity):
        print(f"Entity '{entity}' not found in graph")
        return None
    
    # Get subgraph around the entity
    neighbours = set([entity])
    neighbours.update(list(tg_db.main_graph.successors(entity))[:10])
    neighbours.update(list(tg_db.main_graph.predecessors(entity))[:10])
    
    # Add some second-hop neighbours if we have enough room
    if len(neighbours) < max_nodes:
        for neighbour in list(neighbours)[:5]:
            if len(neighbours) >= max_nodes:
                break
            neighbours.update(list(tg_db.main_graph.successors(neighbour))[:2])
    
    neighbours = list(neighbours)[:max_nodes]
    subgraph = tg_db.main_graph.subgraph(neighbours)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Use NetworkX layout
    pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
    
    # Separate nodes by dataset
    multitq_nodes = []
    timequestions_nodes = []
    unknown_nodes = []
    
    for node in subgraph.nodes():
        dataset = tg_db.main_graph.nodes[node].get('dataset', 'unknown')
        if dataset == 'MultiTQ':
            multitq_nodes.append(node)
        elif dataset == 'TimeQuestions':
            timequestions_nodes.append(node)
        else:
            unknown_nodes.append(node)
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_colour='gray', width=0.5)
    
    # Draw nodes by dataset
    if multitq_nodes:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=multitq_nodes, 
                              node_colour='orange', node_size=300, alpha=0.8, label='MultiTQ')
    
    if timequestions_nodes:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=timequestions_nodes, 
                              node_colour='lightblue', node_size=300, alpha=0.8, label='TimeQuestions')
    
    if unknown_nodes:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=unknown_nodes, 
                              node_colour='lightgray', node_size=300, alpha=0.8, label='Unknown')
    
    # Highlight the central entity
    nx.draw_networkx_nodes(subgraph, pos, nodelist=[entity], 
                          node_colour='red', node_size=500, alpha=0.9)
    
    # Draw labels for important nodes
    important_nodes = [entity] + [n for n in neighbours if subgraph.degree(n) > 2][:8]
    labels = {node: node.replace('_', '\n') if len(node) > 15 else node 
             for node in important_nodes}
    
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, font_weight='bold')
    
    plt.title(f'Subgraph around "{entity}"', fontsize=14, fontweight='bold', pad=20)
    plt.legend(scatterpoints=1, loc='upper right')
    plt.axis('off')
    
    # Add statistics text
    degree = subgraph.degree(entity)
    plt.text(0.02, 0.98, f'Central entity: {entity}\nDegree: {degree}\nSubgraph size: {len(neighbours)} nodes', 
            transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolour='white', alpha=0.8))
    
    # Save the plot
    safe_entity_name = entity.replace('/', '_').replace(' ', '_')
    output_file = f"analysis_results/plots/subgraph_{safe_entity_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Subgraph visualisation saved to: {output_file}")
    plt.close()
    
    return output_file

def create_temporal_activity_plot(tg_db: TemporalGraphDatabase, entity: str):
    """Create temporal activity plot for an entity"""
    
    print(f"Creating temporal activity plot for entity: {entity}")
    
    if not tg_db.main_graph.has_node(entity):
        print(f"Entity '{entity}' not found in graph")
        return None
    
    # Get all edges involving this entity
    entity_edges = []
    
    # Outgoing edges
    for neighbour in tg_db.main_graph.successors(entity):
        for key, data in tg_db.main_graph[entity][neighbour].items():
            entity_edges.append({
                'relation': data.get('relation', 'unknown'),
                'timestamp': data.get('timestamp', 'unknown'),
                'direction': 'outgoing'
            })
    
    # Incoming edges
    for predecessor in tg_db.main_graph.predecessors(entity):
        for key, data in tg_db.main_graph[predecessor][entity].items():
            entity_edges.append({
                'relation': data.get('relation', 'unknown'),
                'timestamp': data.get('timestamp', 'unknown'),
                'direction': 'incoming'
            })
    
    df_edges = pd.DataFrame(entity_edges)
    
    # Extract years
    df_edges['year'] = df_edges['timestamp'].apply(lambda x: 
        int(x.split('-')[0]) if '-' in str(x) and len(str(x).split('-')[0]) == 4 
        else int(str(x)[:4]) if len(str(x)) >= 4 and str(x)[:4].isdigit()
        else None
    )
    
    df_edges = df_edges.dropna(subset=['year'])
    df_edges = df_edges[(df_edges['year'] >= 1990) & (df_edges['year'] <= 2025)]
    
    if len(df_edges) == 0:
        print(f"No valid temporal data found for entity {entity}")
        return None
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Temporal Activity for Entity: {entity}', fontsize=14, fontweight='bold')
    
    # Plot 1: Activity over time
    yearly_activity = df_edges.groupby('year').size().reset_index(name='count')
    
    ax1.plot(yearly_activity['year'], yearly_activity['count'], marker='o', linewidth=2, markersize=4)
    ax1.fill_between(yearly_activity['year'], yearly_activity['count'], alpha=0.3)
    ax1.set_title('Total Activity Over Time', fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Relations')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top relations over time
    top_relations = df_edges['relation'].value_counts().head(5).index
    
    for relation in top_relations:
        rel_data = df_edges[df_edges['relation'] == relation]
        rel_yearly = rel_data.groupby('year').size().reset_index(name='count')
        
        ax2.plot(rel_yearly['year'], rel_yearly['count'], 
                marker='o', linewidth=2, markersize=3, 
                label=relation[:25] + '...' if len(relation) > 25 else relation)
    
    ax2.set_title('Top 5 Relations Over Time', fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Frequency')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    safe_entity_name = entity.replace('/', '_').replace(' ', '_')
    output_file = f"analysis_results/plots/temporal_activity_{safe_entity_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Temporal activity plot saved to: {output_file}")
    plt.close()
    
    return output_file

def create_relation_analysis_plot(tg_db: TemporalGraphDatabase):
    """Create relation analysis plots"""
    
    print("Creating relation analysis plots")
    
    rel_stats = tg_db.get_relation_statistics()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Relation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Top 20 relations
    top_20_relations = list(rel_stats['relation_frequency'].items())[:20]
    relations = [r[0][:15] + '...' if len(r[0]) > 15 else r[0] for r in top_20_relations]
    counts = [r[1] for r in top_20_relations]
    
    bars = ax1.barh(range(len(relations)), counts)
    ax1.set_yticks(range(len(relations)))
    ax1.set_yticklabels(relations, fontsize=7)
    ax1.set_title('Top 20 Relations by Frequency', fontweight='bold')
    ax1.set_xlabel('Frequency')
    
    # 2. Relation frequency distribution
    all_counts = list(rel_stats['relation_frequency'].values())
    ax2.hist(all_counts, bins=50, alpha=0.7, colour='lightgreen', edgecolour='black')
    ax2.set_title('Relation Frequency Distribution', fontweight='bold')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Number of Relations')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Temporal coverage of top relations
    top_10_relations = [r[0] for r in top_20_relations[:10]]
    temporal_coverage = [rel_stats['temporal_coverage'].get(rel, 0) for rel in top_10_relations]
    
    bars = ax3.bar(range(len(top_10_relations)), temporal_coverage, colour='coral')
    ax3.set_xticks(range(len(top_10_relations)))
    ax3.set_xticklabels([r[:10] + '...' if len(r) > 10 else r for r in top_10_relations], 
                       rotation=45, ha='right', fontsize=8)
    ax3.set_title('Temporal Coverage (Unique Timestamps)', fontweight='bold')
    ax3.set_ylabel('Number of Unique Timestamps')
    
    # 4. Dataset comparison
    multitq_relations = defaultdict(int)
    timequestions_relations = defaultdict(int)
    
    for u, v, data in tg_db.main_graph.edges(data=True):
        relation = data.get('relation', 'unknown')
        dataset = data.get('dataset', 'unknown')
        
        if dataset == 'MultiTQ':
            multitq_relations[relation] += 1
        elif dataset == 'TimeQuestions':
            timequestions_relations[relation] += 1
    
    # Get top 10 relations from each dataset
    top_multitq = sorted(multitq_relations.items(), key=lambda x: x[1], reverse=True)[:10]
    top_timequestions = sorted(timequestions_relations.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Find common relations
    multitq_rel_names = [r[0] for r in top_multitq]
    timequestions_rel_names = [r[0] for r in top_timequestions]
    common_relations = list(set(multitq_rel_names[:5]) & set(timequestions_rel_names[:5]))[:5]
    
    if common_relations:
        multitq_counts = [multitq_relations[rel] for rel in common_relations]
        timequestions_counts = [timequestions_relations[rel] for rel in common_relations]
        
        x = np.arange(len(common_relations))
        width = 0.35
        
        ax4.bar(x - width/2, multitq_counts, width, label='MultiTQ', colour='orange', alpha=0.8)
        ax4.bar(x + width/2, timequestions_counts, width, label='TimeQuestions', colour='lightblue', alpha=0.8)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels([r[:10] + '...' if len(r) > 10 else r for r in common_relations], 
                           rotation=45, ha='right', fontsize=8)
        ax4.set_title('Common Relations by Dataset', fontweight='bold')
        ax4.set_ylabel('Frequency')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No common relations\nin top 5 of both datasets', 
                ha='centre', va='centre', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Common Relations by Dataset', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "analysis_results/plots/relation_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Relation analysis saved to: {output_file}")
    plt.close()
    
    return output_file

def main():
    """Main function to create the visualisations"""
    
    print("=== Creating Temporal KG Visuals ===")
    
    # Load the temporal graph database
    tg_db = TemporalGraphDatabase()
    tg_db.load_database()
    
    # Create output directory
    Path("analysis_results").mkdir(exist_ok=True)
    
    # 1. Create overview plots
    overview_file = create_temporal_overview_plots(tg_db)
    
    # 2. Create relation analysis
    relation_file = create_relation_analysis_plot(tg_db)
    
    # 3. Create subgraph plots for key entities
    sample_entities = ['Abdul_Kalam', 'Mahmoud_Abbas']
    
    # Find entities that actually exist
    available_entities = []
    for entity in sample_entities:
        if tg_db.main_graph.has_node(entity):
            available_entities.append(entity)
    
    # If none exist, use most connected entities
    if not available_entities:
        node_degrees = [(node, tg_db.main_graph.degree(node)) 
                       for node in tg_db.main_graph.nodes()]
        node_degrees.sort(key=lambda x: x[1], reverse=True)
        available_entities = [node for node, degree in node_degrees[:2]]
    
    subgraph_files = []
    activity_files = []
    
    for entity in available_entities:
        subgraph_file = create_entity_subgraph_plot(tg_db, entity)
        activity_file = create_temporal_activity_plot(tg_db, entity)
        
        if subgraph_file:
            subgraph_files.append(subgraph_file)
        if activity_file:
            activity_files.append(activity_file)
    
    print(f"\n=== Visuals created ===")
    print(f"Generated PNG files in analysis_results/:")
    print(f"- {overview_file}: Overall KG statistics")
    print(f"- {relation_file}: Relation analysis")
    for file in subgraph_files:
        print(f"- {file}: Entity subgraph")
    for file in activity_files:
        print(f"- {file}: Temporal activity")

if __name__ == "__main__":
    main()
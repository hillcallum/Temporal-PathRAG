#!/bin/bash

# Run the temporal reasoning visualizer
echo "Starting Temporal PathRAG Demo"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed - please run: pip install streamlit"
    exit 1
fi

# Run the demo
streamlit run visualisations/temporal_pathrag_demo.py --server.port 8501 --server.address localhost
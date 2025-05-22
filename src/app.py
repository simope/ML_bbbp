import streamlit as st
from utils import visualize_molecule_3d
from predict import predict
import graphviz
import torch
from models.neural_network import BbbpNN
import config

# Set page title
st.title("Brain Blood-Barrier Penetration prediction")

# Add a text input for SMILES
smiles = st.text_input("Enter SMILES string:", "CC(=O)OC1=CC=CC=C1C(=O)O")

# Create two columns
col1, col2 = st.columns(2)

# Left column: Molecule visualization
with col1:
    st.subheader("3D Molecule View")
    try:
        html = visualize_molecule_3d(smiles)
        st.components.v1.html(html, height=400)
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Right column: Model predictions
with col2:
    st.subheader("Is it BBBP?")
    try:
        # Get prediction
        penetrating = predict(smiles)
        label = "can" if penetrating else "cannot"
        emoji = "✅" if penetrating else "❌"
        # Display prediction with a progress bar
        message = f"""{emoji} According to this model's prediction,
        this molecule {label} pass through the BBB."""
        st.write(message)
            
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")


# Add explanation section
with st.expander("About this Project", expanded=False):
    st.markdown("""
    ### What is BBBP?
    The Blood-Brain Barrier (BBB) is a highly selective membrane that separates the circulating blood from the brain's extracellular fluid. 
    BBBP (Blood-Brain Barrier Penetration) prediction is crucial in drug discovery as it helps determine whether a molecule can effectively reach the brain.

    ### How does it work?
    In this learning project I used a neural network trained on molecular data to predict whether a given molecule can penetrate the blood-brain barrier. 
    The model analyzes the molecular structure (represented as SMILES strings), converting it into Morgan fingerprints, and makes predictions based on learned patterns from known BBB-penetrating compounds.

    ### How to use:
    1. Enter a SMILES string in the input box
    2. View the 3D molecular structure on the left
    3. Get the BBB penetration prediction on the right
    """)

# Add model architecture section
with st.expander("Model Architecture", expanded=False):
    st.markdown("""
    ### Neural Network Architecture
    The model I used is a neural network with the following architecture:
    """)
    
    # Create a simple visualization of the model architecture
    dot = graphviz.Digraph(comment='Model Architecture')
    dot.body.append('graph [bgcolor="#0E1117", pad="0.5", splines="ortho", nodesep="0.5", ranksep="0.5"];')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    dot.attr('edge', color='#FFFFFF')  # Light gray
    
    # Initialize model and get layer sizes
    model = BbbpNN()
    try:
        model.load_state_dict(torch.load(config.NN_PATH))
    except:
        st.info("Using untrained model for visualization")
    
    # Extract layer sizes from the model
    layer_sizes = []
    for layer in model.model:
        if isinstance(layer, torch.nn.Linear):
            layer_sizes.append(layer.in_features)
    layer_sizes.append(1)  # Add output layer size
    

    # Add nodes with layer information
    dot.node("smiles", "SMILES string\n(however long)")
    dot.node('input', f'Input Layer\n{layer_sizes[0]} neurons\n(Morgan fingerprints)')
    dot.node('hidden1', f'Hidden Layer 1\n{layer_sizes[1]} neurons\nReLU + Dropout(0.3)')
    dot.node('hidden2', f'Hidden Layer 2\n{layer_sizes[2]} neurons\nReLU + Dropout(0.3)')
    dot.node('output', f'Output Layer\n{layer_sizes[-1]} neuron\nSigmoid')
    
    # Add edges
    dot.edge('smiles', 'input')
    dot.edge('input', 'hidden1')
    dot.edge('hidden1', 'hidden2')
    dot.edge('hidden2', 'output')
    
    # Center the graph using columns
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.graphviz_chart(dot.source, use_container_width=True)
    
    # Add model details
    st.markdown("""
    ### Model Details
    The model processes molecular fingerprints through the following layers:
    
    1. **Input Layer ({} neurons)**
       - Takes Morgan fingerprints as input
       - Each neuron represents a specific molecular feature
    
    2. **Hidden Layer 1 ({} neurons)**
       - ReLU activation function
       - 30% dropout for regularization
       - Learns complex molecular patterns
    
    3. **Hidden Layer 2 ({} neurons)**
       - ReLU activation function
       - 30% dropout for regularization
       - Further refines molecular patterns
    
    4. **Output Layer ({} neuron)**
       - Sigmoid activation function
       - Produces probability of BBB penetration
    """.format(layer_sizes[0], layer_sizes[1], layer_sizes[1], layer_sizes[-1]))

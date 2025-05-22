import streamlit as st
from utils import visualize_molecule_3d
from predict import predict

# Set page title
st.header("Brain Blood-Barrier Penetration prediction", )

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
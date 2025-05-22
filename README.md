# BBBP Neural Network Project

A machine learning project for predicting Blood-Brain Barrier Penetration (BBBP) using neural networks. This project combines cheminformatics and deep learning to predict whether a molecule can penetrate the blood-brain barrier, which is crucial for drug development.

## Live Demo

Try the model yourself at: [BBBP Predictor App](https://bbbppredict.streamlit.app/)

The web application allows you to:
- Input any molecule using its SMILES string
- View a 3D visualization of the molecule
- Get an instant prediction of BBB penetration
- Explore the model architecture and how it works

## What is BBBP?

The Blood-Brain Barrier (BBB) is a highly selective membrane that separates the circulating blood from the brain's extracellular fluid. It acts as a protective barrier, allowing only specific molecules to pass through while blocking others. Predicting BBB penetration is crucial for:
- Drug development for neurological diseases
- Understanding drug delivery to the brain
- Assessing potential side effects of new compounds

## Project Structure

```
BBBP/
├── data/               # Dataset and processed data
├── models/            # Neural network model implementation
├── notebooks/         # Jupyter notebooks for development
├── src/              # Source code
│   ├── app.py        # Streamlit web application
│   ├── config.py     # Configurations
│   ├── train.py      # Training functions
│   ├── predict.py    # Prediction functions
│   ├── preprocess.py # Data preprocessing
│   └── utils.py      # Utility functions
└── README.md         # This file
```

## Model Architecture

The project uses a neural network with the following architecture:
- Input Layer: 2048 neurons (Morgan fingerprints)
- Hidden Layer 1: 128 neurons with ReLU activation and dropout
- Hidden Layer 2: 128 neurons with ReLU activation and dropout
- Output Layer: 1 neuron with sigmoid activation

The model processes molecular structures by:
1. Converting SMILES strings to Morgan fingerprints
2. Processing these fingerprints through the neural network
3. Outputting a probability of BBB penetration

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BBBP.git
cd BBBP
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit app locally:
```bash
streamlit run src/app.py
```

## Project Goals

- Develop an accurate BBBP prediction model
- Create an accessible web interface for predictions
- Provide educational insights into molecular properties affecting BBB penetration
- Demonstrate the application of machine learning in drug discovery

## Future Improvements

- Add more molecular descriptors
- Implement additional machine learning models
- Include confidence scores for predictions
- Enhance the visualization of molecular properties

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

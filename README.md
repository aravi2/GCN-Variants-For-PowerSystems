# GCN-Variants-For-PowerSystems

# Impedance-Aware Graph Convolutional Network (IA-GCN) for Power Systems
The Impedance-Aware Graph Convolutional Network (IA-GCN) presents an innovative approach to voltage estimation in electrical distribution networks by integrating graph convolutional networks (GCNs) with impedance data. This repository hosts the IA-GCN model, designed to enhance the accuracy of voltage estimation across electrical grids.

# Introduction
Effective voltage estimation is vital for the operational efficiency, control, and reliability of electrical distribution networks. Traditional methodologies often overlook the nuanced relationships within the network's topology. Addressing this gap, the IA-GCN model incorporates impedance magnitudes into the graph convolutional framework, enriching the model's understanding of the network's electrical properties and improving voltage estimation accuracy.

# Getting Started
This section provides guidance on setting up and running the IA-GCN model on your system.

# Prerequisites
Ensure the following tools and libraries are installed:

Python 3.x
PyTorch
PyTorch Geometric
h5py
NumPy
scikit-learn
Matplotlib


# Installation
To get started, clone the repository and install the necessary dependencies:
git clone https://github.com/aravi2/GCN-Variants-For-PowerSystems.git
cd GCN-Variants-For-PowerSystems
pip install torch torchvision torchaudio
pip install torch-geometric
pip install h5py numpy scikit-learn matplotlib


# Data Preparation
Download mydata.hf.zip from the repository, then unzip it.
Move the mydata.h5 file into the project's root directory.


# Running the Model
Execute the Impedance_Mag_Aware_GCN.py script to run the model:
python Impedance_Mag_Aware_GCN.py

This initiates the model's training phase, outputs performance metrics, and generates comparative plots of the predicted versus actual voltage values.

# Repository Overview
Impedance_Mag_Aware_GCN.py: Contains the IA-GCN model implementation, emphasizing impedance modulation within the GCN framework.
mydata.hf.zip: The compressed data file for model input.
Performance
The IA-GCN model markedly improves voltage estimation in electrical distribution networks, surpassing traditional GCN models in precision. Detailed performance metrics are available in the repository.

# Citing This Work
If you use this code in your research, please cite the following paper:
"Impedance-Aware Graph Convolutional Networks for Voltage Estimation in Active Distribution Networks."


Contributions
We welcome contributions to enhance or extend the IA-GCN model. Fork this repository and submit pull requests with your improvements.

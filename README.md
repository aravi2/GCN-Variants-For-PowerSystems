# GCN-Variants-For-PowerSystems

Impedance-Aware Graph Convolutional Network (IA-GCN) for Power Systems
The Impedance-Aware Graph Convolutional Network (IA-GCN) introduces an innovative approach to voltage estimation in electrical distribution networks by leveraging the unique properties of graph convolutional networks (GCNs) and incorporating impedance information directly into the network analysis. This repository contains the implementation of the IA-GCN model, aiming to enhance the accuracy and reliability of voltage estimation tasks.

Introduction
Voltage estimation is a cornerstone of maintaining the efficiency, control, and reliability of electrical distribution networks. Conventional methods often overlook the complex interactions within the network's topology. The IA-GCN model addresses this gap by integrating the magnitude of impedance into the graph convolution process, thus providing a more detailed understanding of the network's electrical behavior.

Getting Started
These instructions will guide you through setting up and running the IA-GCN model on your local machine.

Prerequisites
Ensure you have the following software and libraries installed:

Python 3.x
PyTorch
PyTorch Geometric
h5py
NumPy
scikit-learn
Matplotlib
Installation
Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/aravi2/GCN-Variants-For-PowerSystems.git
cd GCN-Variants-For-PowerSystems
Install the required dependencies:

bash
Copy code
pip install torch torchvision torchaudio
pip install torch-geometric
pip install h5py numpy scikit-learn matplotlib
Data Preparation
Download the mydata.h5 file provided in the repository and unzip it.
Ensure the mydata.h5 file is placed in the root directory of the project.
Running the Model
To execute the IA-GCN model, run the following command:

bash
Copy code
python main.py
This will initiate the training process and display the performance metrics, along with visualizations of predicted vs. actual voltage values.

Repository Structure
ImpedanceAwareGCNConv: The custom Graph Convolutional Network layer that incorporates impedance modulation.
ImpedanceAwareGCN: The full IA-GCN model utilizing the custom layer for voltage estimation tasks.
Performance
The IA-GCN model demonstrates superior performance in estimating voltages within electrical distribution networks, significantly outperforming conventional GCN models. Detailed performance metrics are provided within the repository.

Contribution
Contributions to enhance or extend the IA-GCN model are welcome. Feel free to fork this repository and submit pull requests with your improvements.



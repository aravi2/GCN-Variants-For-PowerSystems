# GCN-Variants-For-PowerSystems
Impedance-Aware Graph Convolutional Network (IA-GCN)
The Impedance-Aware Graph Convolutional Network (IA-GCN) is a novel architecture designed to enhance voltage estimation in electrical distribution networks. By integrating the impedance characteristics directly into the graph convolution process, IA-GCN provides a more nuanced understanding of the network's topology and electrical properties, leading to more accurate predictions.

Introduction
Voltage estimation is crucial for the efficient and reliable operation of electrical distribution networks. Traditional approaches often overlook the complex interplay between network topology and line impedances. The IA-GCN model addresses this gap by incorporating impedance information, improving the accuracy of voltage estimation tasks.

Getting Started
These instructions will guide you through setting up and running the IA-GCN model on your local machine for development and testing purposes.

Prerequisites
Ensure you have the following installed:

Python 3.x
PyTorch
PyTorch Geometric
h5py
NumPy
scikit-learn
Matplotlib
Installing
First, clone the repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/ia-gcn.git
cd ia-gcn
Next, install the required Python packages:

bash
Copy code
pip install torch torchvision torchaudio
pip install torch-geometric
pip install h5py numpy scikit-learn matplotlib
Data Preparation
Download the relevant data file provided in the repository and unzip it.
Place the unzipped data file (mydata.h5) in the root directory of the project.
Running the Code
To run the IA-GCN model, execute the main script:

bash
Copy code
python main.py
This script will load the data, train the IA-GCN model, and output the performance metrics along with plots illustrating the predicted vs. actual voltage values.

Code Overview
The repository contains two main components:

ImpedanceAwareGCNConv: A custom Graph Convolutional Network layer that incorporates impedance modulation.
ImpedanceAwareGCN: The full model that utilizes the custom GCN layer for voltage estimation.
Performance
The IA-GCN model demonstrates superior performance in voltage estimation tasks, significantly reducing error metrics compared to conventional GCN models and providing a deep insight into the network's behavior.

Contributing
Contributions to improve the IA-GCN model or extend its applications are welcome. Please feel free to fork the repository and submit pull requests.

License

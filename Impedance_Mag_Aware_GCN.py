import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import MessagePassing
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Define ImpedanceAwareGCNConv
class ImpedanceAwareGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ImpedanceAwareGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.weight_impedance = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.weight_impedance)

    def forward(self, x, edge_index, impedance):
        out = self.lin(x)
        impedance_modulation = x @ self.weight_impedance
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=out, impedance_modulation=impedance_modulation, impedance=impedance)

    def message(self, x_j, edge_index, impedance_modulation, impedance, size):
        row, col = edge_index
        impedance_modulation = impedance_modulation[col, :]
        modulation = impedance[row] * impedance_modulation
        return x_j + modulation

    def update(self, aggr_out):
        return aggr_out


# Update the GCN model to use ImpedanceAwareGCNConv
class ImpedanceAwareGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, beta):
        super(ImpedanceAwareGCN, self).__init__()
        self.beta = beta
        self.conv1 = ImpedanceAwareGCNConv(input_dim, hidden_dim1)
        self.conv2 = ImpedanceAwareGCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = ImpedanceAwareGCNConv(hidden_dim2, output_dim)
        self.fc = torch.nn.Linear(hidden_dim2, 1)  # Fully connected layer to get a single output value

    def forward(self, data):
        x, edge_index, impedance = data.x, data.edge_index, data.impedance
        x = self.conv1(x, edge_index, impedance)
        x = F.leaky_relu(x, negative_slope=self.beta)
        x = self.conv2(x, edge_index, impedance)
        x = F.leaky_relu(x, negative_slope=self.beta)
        x = self.conv3(x, edge_index, impedance)
        x = self.fc(x)  # Pass through fully connected layer
        return x


# Load MATLAB data saved in HDF5 format
with h5py.File('mydata.h5', 'r') as file:
    P_inj_data = torch.tensor(file['P_inj_data'][:], dtype=torch.float32)
    Q_inj_data = torch.tensor(file['Q_inj_data'][:], dtype=torch.float32)
    DER_Status_data = torch.tensor(file['DER_Status_data'][:], dtype=torch.float32)
    
    # Here, only load the resistance data
    Resistance_data = torch.tensor(file['Impedance_data_real'][:], dtype=torch.float32)
    Reactance_data = torch.tensor(file['Impedance_data_imag'][:], dtype=torch.float32)
    V_label_data = torch.tensor(file['V_label_data'][:], dtype=torch.float32)
    # Calculate impedance magnitude
    Impedance_data = torch.sqrt(Resistance_data ** 2 + Reactance_data ** 2)

# Perform zero-mean normalization (Z-score normalization) for the input features
P_inj_data = (P_inj_data - P_inj_data.mean()) / P_inj_data.std()
Q_inj_data = (Q_inj_data - Q_inj_data.mean()) / Q_inj_data.std()
#V_label_data = (V_label_data - V_label_data.mean()) / V_label_data.std()

# Assuming the first dimension is the number of graphs.
num_graphs = P_inj_data.size(0)
data_list = []
for i in range(num_graphs):
    for t in range(P_inj_data.size(1)):
        x = torch.stack([P_inj_data[i,t,:], Q_inj_data[i,t,:], DER_Status_data[i,t,:]], dim=1)
        impedance = Impedance_data[i]
        edge_index = impedance.nonzero(as_tuple=False).t()
        y = V_label_data[i,t,:]
        data = Data(x=x, edge_index=edge_index, impedance=impedance, y=y)  # Include impedance data
        data_list.append(data)

# Split the data into training, validation, and testing sets (e.g., 60% training, 20% validation, 20% testing)
train_data, temp_data = train_test_split(data_list, test_size=0.4, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Define lists to hold the training and validation losses for each epoch
train_loss_history = []
val_loss_history = []

# Create DataLoaders for each dataset with batch size of 1024
train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=1024, shuffle=False)
test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)

# Define dimensions for the GCN layers
input_dim = 3
hidden_dim1 = 33
hidden_dim2 = 33
# hidden_dim1 = 59#
# hidden_dim2 = 62
output_dim = 1

# Create the GCN model, define the loss function and optimizer
# Create the GCN model, define the loss function and optimizer
model = ImpedanceAwareGCN(input_dim, hidden_dim1, hidden_dim2, output_dim, beta=0.1)
#model = GCN(input_dim, hidden_dim1, hidden_dim2, output_dim, beta=0.4)
loss_criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.00584791383967731)

# Set number of epochs and learning rate decay coefficient
num_epochs = 100
lr_delay_coefficient = 0.99

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    # For each batch in the training data, perform forward pass, calculate loss, perform backpropagation and optimize
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_criterion(out.squeeze(), batch.y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    # Print training loss for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss / len(train_loader)}")
    
    model.eval()
    total_val_loss = 0.0
    # Calculate validation loss
    with torch.no_grad():
        for batch in valid_loader:
            out = model(batch)
            val_loss = loss_criterion(out.squeeze(), batch.y)
            total_val_loss += val_loss.item()
    
    # Print validation loss for each epoch
    avg_val_loss = total_val_loss / len(valid_loader)
    
    # Print validation loss for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}")

    # Stop training if the average validation loss is below the threshold
    if avg_val_loss <= .5e-6:
        print("Validation loss reached below 1e-7. Stopping training.")
        break

    # Append the average loss values to the lists
    train_loss_history.append(total_loss / len(train_loader))
    val_loss_history.append(total_val_loss / len(valid_loader))
    
    # Decay the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_delay_coefficient

print("Training completed!")

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Evaluate the model on the test dataset
model.eval()
predictions = []
actual_values = []

# Get predictions and actual values for the test set
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        predictions.extend(out.numpy().flatten())
        actual_values.extend(batch.y.numpy().flatten())

# Determine the number of nodes and time periods from the dimensions of the input data
num_nodes = P_inj_data.size(2)
num_time_periods = len(predictions) // num_nodes

# Reshape the predictions and actual values into matrices
predictions = np.array(predictions).reshape(num_time_periods, num_nodes)
actual_values = np.array(actual_values).reshape(num_time_periods, num_nodes)

# Compute MAE, MSE, RMSE, R2, explained variance, and MAPE
mae = mean_absolute_error(actual_values, predictions)
mse = mean_squared_error(actual_values, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual_values, predictions)
explained_variance = explained_variance_score(actual_values, predictions)
mape = np.mean(np.abs((np.array(actual_values) - np.array(predictions)) / np.array(actual_values))) * 100

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared: {r2}')
print(f'Explained Variance: {explained_variance}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

# Scatter plot of Predicted vs Actual values
plt.figure(figsize=(8, 8))
plt.scatter(actual_values, predictions, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

# Residual Plot
residuals = np.array(actual_values) - np.array(predictions)
plt.figure(figsize=(8, 8))
plt.scatter(predictions, residuals, alpha=0.5)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Select the portion of the data that corresponds to the first 24 time periods
predictions_24h = predictions[:24, :]
actual_values_24h = actual_values[:24, :]


# Plot the predicted and actual values for all nodes across the first 24 time periods
plt.figure(figsize=(15, 7))
for i in range(num_nodes):
    plt.plot(predictions_24h[:, i], linestyle='dashed', linewidth=1)
    plt.plot(actual_values_24h[:, i], linestyle='solid', linewidth=1)

plt.xlabel('Time Period (Horizons)')
plt.ylabel('Voltage')
plt.title('Predicted vs Actual Values for All 33 Nodes Over 24 Horizons')
plt.legend(['Predicted', 'Actual'])
plt.show()
import torch 
import torch.nn as nn
import pennylane as qml

n_qubits=4
n_layers=3
quantum_device = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(quantum_device, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
            
        for i in range(n_qubits):
            control_wire = i
            target_wire = (i + 1) % n_qubits
            qml.CRZ(weights[layer, i], wires=[control_wire, target_wire])

    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HQNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc_to_four_dim = nn.Linear(32 * 2 * 2, 4)

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        self.post_net = nn.Linear(n_qubits, num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_to_four_dim(x)
        x = self.quantum_layer(x)
        x = self.post_net(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    model = HQNNClassifier(num_classes=5)
    print(model)

    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)

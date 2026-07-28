import torch 
import torch.nn as nn
import pennylane as qml

n_qubits = 4
quantum_device = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(quantum_device, interface="torch")
def quantum_circuit(inputs, weight):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    
    for layer in range(n_qubits - 1):
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
            
        for i in range(n_qubits):
            control_wire = i
            target_wire = (i + 1) % n_qubits
            qml.CRZ(weight[layer, i], wires=[control_wire, target_wire])

    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

if __name__ == "__main__":
    sample_inputs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    sample_weights = torch.tensor([
        [0.5, 0.6, 0.7, 0.8],  # Tầng 1
        [0.1, 0.2, 0.3, 0.4],  # Tầng 2
        [0.9, 1.0, 1.1, 1.2]   # Tầng 3
    ])

    print("--- Cấu trúc mạch lượng tử biểu diễn theo sơ đồ Fig. 8 ---")
    print(qml.draw(quantum_circuit)(inputs=sample_inputs, weight=sample_weights))
    
    print("\n--- Kết quả đầu ra sau phép đo lượng tử ---")
    output = quantum_circuit(inputs=sample_inputs, weight=sample_weights)
    print([val.item() for val in output])
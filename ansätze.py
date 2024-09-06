import sys
import os
from typing import Any, Literal, Callable, Iterable, Dict
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
# from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp
# from qiskit.circuit.library import TwoLocal
import numpy as np
# import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from qiskit.visualization import plot_state_hinton


def ansatz_simple_rotation(n_qubits: int, axis: str = 'x', save_state = True):
    
    if axis not in ('x', 'y', 'z'):
        raise ValueError("El argumento 'axis' debe ser un string válido")
    
    theta = ParameterVector("theta", n_qubits)

    qc = QuantumCircuit(n_qubits,n_qubits)

    if axis == 'x' or axis == 'X':
        for i in range(n_qubits):
            qc.rx(theta[i], i)
    elif axis == 'y' or axis == 'Y':
        for i in range(n_qubits):
            qc.ry(theta[i], i)
    elif axis == 'z' or axis == 'Z':
        for i in range(n_qubits):
            qc.rz(theta[i], i)
    else:
        print("Axis must be x/y/z or X/Y/Z")
        sys.exit(1)

    if save_state:
        qc.save_statevector()

    return qc


def ansatz_Efficient_SU2(n_qubits: int, save_state = True):

    theta = ParameterVector("theta", 4 * n_qubits)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    for i in range(n_qubits):
        qc.ry(theta[i], i)
        qc.rz(theta[n_qubits + i],i)
    qc.barrier()
    for i in range(n_qubits-1):
        qc.cx(i, i+1)
    qc.barrier()
    for i in range(n_qubits):
        qc.ry(theta[2*n_qubits + i], i)
        qc.rz(theta[3*n_qubits + i],i)

    if save_state:
        qc.save_statevector()

    return qc

EntanglementNames = Literal['linear', 'full', 'circular', 'reverse_linear']
EntanglementGateNames = Literal['cx', 'crx', 'cry', 'crz']
RotationGates = Literal['rx', 'ry', 'rz']
rotation_gate_dict: Dict[str, Any] = {
    'rx': QuantumCircuit.rx,
    'ry': QuantumCircuit.ry,
    'rz': QuantumCircuit.rz
}
entanglement_gate_dict: Dict[str, Any] = {
    'crx': QuantumCircuit.crx,
    'cry': QuantumCircuit.cry,
    'crz': QuantumCircuit.crz
}

def ansatz_two_local(n_qubits: int, repetitions: int = 1, rotation_gate: RotationGates = 'ry',
                      entanglement: EntanglementNames = 'linear', entanglement_gate: EntanglementGateNames = 'cx', save_state = True):
    
    if entanglement_gate in ('crx', 'cry', 'crz'):
        if entanglement in ('linear', 'reverse_linear'):
            num_params = n_qubits * (2 * repetitions + 1) - repetitions
        elif entanglement == 'full':
            num_params = n_qubits * ( (n_qubits-1)*repetitions/2 + repetitions+1 )
        elif entanglement == 'circular':
            num_params = n_qubits * ( 2*repetitions + 1)
    elif entanglement_gate in ('cx',):
        num_params = n_qubits*(repetitions+1)
    
    theta = ParameterVector(name = 'theta', length = num_params)

    # if entanglement_gate in EntanglementGateNames:
    #     theta = ParameterVector(name = 'theta', length = n_qubits*(2*repetitions+1)-repetitions) # n_qubits*(repetitions+1) + (n_qubits-1)*repetitions
    # else:
    #     theta = ParameterVector(name = "theta", length = n_qubits*(repetitions+1))

    qc = QuantumCircuit(n_qubits, n_qubits)

    theta_index = 0

    for k in range(repetitions):
        for i in range(n_qubits):
            rotation_gate_dict[rotation_gate](qc, theta[theta_index], i)
            # qc.ry(theta[theta_index], i)
            theta_index+= 1
        qc.barrier()

        if entanglement == 'linear':
            for i in range(n_qubits-1):
                if entanglement_gate == 'cx':
                    qc.cx(i, i+1)
                else:
                    entanglement_gate_dict[entanglement_gate](qc,theta[theta_index], i, i+1)
                    theta_index += 1
                    

        elif entanglement == 'full':
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    if entanglement_gate == 'cx':
                        qc.cx(i, j)
                    else:
                        entanglement_gate_dict[entanglement_gate](qc,theta[theta_index], i, j)
                        theta_index += 1

        elif entanglement == 'circular':
            for i in range(n_qubits-1):
                if entanglement_gate == 'cx':
                    qc.cx(i, i+1)
                else:
                    entanglement_gate_dict[entanglement_gate](qc,theta[theta_index], i, i+1)
                    theta_index += 1
            if entanglement_gate == 'cx':
                qc.cx(n_qubits-1, 0)
            else:
                entanglement_gate_dict[entanglement_gate](qc,theta[theta_index], n_qubits-1, 0)
                theta_index += 1

        elif entanglement == 'reverse_linear':
            for i in range(n_qubits-2, -1, -1):
                if entanglement_gate == 'cx':
                    qc.cx(i, i+1)
                else:
                    entanglement_gate_dict[entanglement_gate](qc,theta[theta_index], i, i+1)
                    theta_index += 1

        qc.barrier()
    for i in range(n_qubits):
        rotation_gate_dict[rotation_gate](qc, theta[theta_index], i)
        theta_index += 1

    if save_state:
        qc.save_statevector()

    return qc

ParamSetType = Literal['unique', 'complete']

def ansatz_transerse_ising(n_qubits: int, rotation_gate: RotationGates = 'ry', param_set: ParamSetType = 'unique', save_state = True):
    
    qc = QuantumCircuit(n_qubits, n_qubits)

    if param_set == 'unique':
        num_params = 3
    elif param_set == 'complete':
        num_params = n_qubits + 2

    theta = ParameterVector(name = 'theta', length = num_params)

    qc.rz(theta[0], 0)
    qc.ry(theta[1], 0)

    qc.barrier()

    for i in range(1, n_qubits):
        qc.cx(0, i)
    
    qc.barrier()

    for i in range(n_qubits):
        if param_set == 'unique':
            rotation_gate_dict[rotation_gate](qc, theta[2], i)
        elif param_set == 'complete':
            rotation_gate_dict[rotation_gate](qc, theta[i+2], i)

    if save_state:
        qc.save_statevector()

    return qc



AnsatzName = Literal['ansatz_simple_rotation', 'ansatz_Efficient_SU2', 'ansatz_two_local', 'ansatz_transerse_ising']

ansatz_functions = {
    'ansatz_simple_rotation': ansatz_simple_rotation,
    'ansatz_Efficient_SU2': ansatz_Efficient_SU2,
    'ansatz_two_local': ansatz_two_local,
    'ansatz_transerse_ising': ansatz_transerse_ising
}
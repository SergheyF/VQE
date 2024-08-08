import sys
import os
from typing import Callable, Literal, Dict, Any, Tuple, List, Iterable
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal
#from qiskit.utils import algorithm_globals
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from qiskit.visualization import plot_state_hinton 
from modules_v1 import ansätze, hamiltonians, utils

ErrorNames = Literal['bit-flip']

class Vqe:
    
    def __init__(self,n_qubits: int, error_type: ErrorNames = None, **kwargs) -> None:
        # El error se debe especificar como una tupla o lista [RESET, MEAS, 1GATE, 2GATE]
        self.n_qubits = n_qubits
        self.noise_model = None

        if error_type:
            self.noise_model = NoiseModel()

            if error_type == 'bit-flip':
                probs = kwargs.get('probs', [0.05, 0.05, 0.05, 0.05])
                if probs[0] != 0:
                    error_reset = pauli_error([('X', probs[0]), ('I', 1 - probs[0])])
                    self.noise_model.add_all_qubit_quantum_error(error = error_reset, instructions = 'reset')

                if probs[1] != 0:
                    error_meas = pauli_error([('X', probs[1]), ('I', 1 - probs[1])])
                    self.noise_model.add_all_qubit_quantum_error(error_meas, "measure")

                if probs[2] != 0:
                    error_gate1 = pauli_error([('X', probs[2]), ('I', 1 - probs[2])])
                    self.noise_model.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])

                if probs[3] != 0:
                    error_gate2 = pauli_error([('X', probs[3]), ('I', 1 - probs[3])])
                    error_gate2 = error_gate2.tensor(error_gate2)
                    self.noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"])

    def __call__(self, hamiltoniano: dict, num_iterations: int = 1, ansatz_name: ansätze.AnsatzName = None, theta_values: Iterable = None, **ansatz_kwargs: Any) -> Tuple[Dict, float]:

        n_qubits = self.n_qubits
        start = time.perf_counter()
        glob_runtime: float = 0.0

        if ansatz_name is not None:
            ansatz_fun = ansätze.ansatz_functions[ansatz_name]
            ansatz = ansatz_fun(n_qubits = n_qubits, **ansatz_kwargs)
        else:
            print("Es necesario expecificar un tipo de ansatz")
            sys.exit(1)
        
        expected_value_H_ising_1D = utils.ExpectedValueHIsing1DWrapper(noise_model = self.noise_model)
        cost_function = lambda p: expected_value_H_ising_1D(p, h_terms = hamiltoniano, ansatz = ansatz)
        magnetizacion = hamiltonians.ising_Hamiltonian_1D(n_qubits = n_qubits,J1 = (-1,) , scope = 0)

        if theta_values is None:
            aleatory = True
        else:
            aleatory = False

        rng = np.random.default_rng()

        glob_history = []

        cycle = 0

        for _ in range(num_iterations):
            
            if aleatory and ansatz_name == 'ansatz_simple_rotation':
                theta_values = rng.random(n_qubits) * 4 * np.pi
            elif aleatory and ansatz_name == 'ansatz_Efficient_SU2':
                theta_values = rng.random(4 * n_qubits) * 4 * np.pi
            elif aleatory and ansatz_name == 'ansatz_two_local':
                rep = ansatz_kwargs.get('repetitions', 1)
                entanglement = ansatz_kwargs.get('entanglement', 'linear')
                entanglement_gate = ansatz_kwargs.get('entanglement_gate', 'cx')
                if entanglement_gate in ('crx', 'cry', 'crz'):
                    if entanglement in ('linear', 'reverse_linear'):
                        theta_values = rng.random(n_qubits * (2 * rep + 1) - rep) * 4 * np.pi
                    elif entanglement == 'full':
                        theta_values = rng.random(n_qubits * ( (n_qubits-1)*rep/2 + rep+1 )) * 4 * np.pi
                    elif entanglement == 'circular':
                        theta_values = rng.random(n_qubits * ( 2*rep + 1)) * 4 * np.pi
                elif entanglement_gate in ('cx',):
                    theta_values = rng.random(n_qubits*(rep+1)) * 4 * np.pi
            elif aleatory and ansatz_name == 'ansatz_transerse_ising':
                param_set = ansatz_kwargs.get('param_set', 'unique')
                if param_set == 'unique':
                    theta_values = rng.random(3) * 4 * np.pi
                elif param_set == 'complete':
                    theta_values = rng.random(n_qubits+2) * 4 * np.pi
                    
            history = {
                'cost': [],
                'params': [],
                'state': []
            }
            def callback(p):
                history["cost"].append(expected_value_H_ising_1D.value)
                history["params"].append(p)
                history["state"].append(expected_value_H_ising_1D.state)

            result_scipy = minimize(fun = cost_function, x0 = theta_values, method = 'COBYLA', callback = callback)
            cycle +=1
            print('\n----------------------------------------------------------------------------------------------------------------------\n')
            print("Cycle:", cycle)
            print("Success:", result_scipy.success)
            print("Status:", result_scipy.status)
            print("Message:", result_scipy.message)
            print("Function value at minimum:", result_scipy.fun)
            print("Optimal parameters:", result_scipy.x)
            print("Number of iterations performed:", result_scipy.get('nit', 'Not available'))
            print("Number of function evaluations:", result_scipy.get('nfev', 'Not available'))
            
            history['number_evaluations'] = result_scipy.get('nfev', 'Not available')
            history['optimal parameters'] = result_scipy.x
            history['iterations'] = result_scipy.get('nit', 'Not available')


            final_circuit = ansatz.assign_parameters(result_scipy.x, inplace = False)
            final_circuit.measure(range(n_qubits), range(n_qubits))
            result, probs = utils.computar_circuito(circuit=final_circuit)
            state = result.get_statevector()
            print(f'La función de onda final tras la optimización es: {state.data}')
            print(f'El resultado de computar el circuito optimizado es (tras normalizarlo): {probs}')
            print(f'El Hamiltoniano analizado es: {hamiltoniano}, con un valor esperado de {result_scipy.fun}')

            ############# MAGNETIZACION ##############

            magnet_avg = expected_value_H_ising_1D(theta_values = history['optimal parameters'], h_terms = magnetizacion, ansatz = ansatz, magn = True)/n_qubits
            print(f'La magnetización es de: {magnet_avg}')
            
            glob_history.append((result, final_circuit, history, cycle, magnet_avg))
            glob_history.sort(key=lambda x: x[2]['cost'][-1])
            # for histories in glob_history:
        
        end = time.perf_counter()
        glob_runtime = end - start


        return glob_history, glob_runtime


def analizar_vqe(job: tuple, hamiltoniano_matriz, ansatz: str, optimizer: str = 'COBYLA', title: str = 'Ising Hamiltonian'):
    glob_history = job[0]
    glob_runtime = job[1]
    horas = glob_runtime // 3600
    minutos = ( glob_runtime % 3600 ) // 60
    segundos = glob_runtime % 60
    n_qubits = int(np.log2(hamiltoniano_matriz.shape[0]))

    # Bloque de diagonalización
    e_ground, (w,v), degeneration = diagonalizar(hamiltoniano_matriz)
    ground_states = v[:, np.where(w == np.min(w))]

    print(f'La matriz del Hamiltoniano es:\n {hamiltoniano_matriz}\n\n'
        'Este hamiltoniano tiene los siguientes estados propios con sus correspondientes energías (calculadas de manera exacta):\n')

    for i in range(len(w)):
        print(f'Autovalor {i+1}: {w[i]}')
        print(f'Autovector {i+1}: {v[:, i]}\n')
        
    print(f'El ground state (exacto) asociado al Hamiltoniano del problema es: \n {ground_states.T} \n'
        f'con una energía de: {e_ground}')
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))

    # Estructura glob_history: List[Tuple[Any, Any, Dict]] = [] 
    # glob_history: List[Tuple[result, final_circuit, history]] = [] 

    for num, iteration in enumerate(glob_history):

        history = iteration[2]
        cycle = iteration[3]
        magnet_avg = iteration[4]
        overlap = []

        for ground_state in ground_states.T:
            overlap_row = []
            for state_vqe in history['state']:
                    overlap_row.append(np.abs(np.vdot(ground_state,state_vqe.data))**2)
            overlap.append(overlap_row)
        total_overlap = [sum(overlaps) for overlaps in zip(*overlap)]
        
        print(f'Total overlap: {total_overlap}')
        # Pintar la primera curva en el primer subplot
        ax[0].plot(history['cost'], label=f'(cycle: {cycle}): {history["cost"][-1]:.4f}\n'
                   f'm_z: {magnet_avg:.2f}', linestyle='-', linewidth=1)

        # Pintar la segunda curva en el segundo subplot
        # for i in overlap:
        
        ax[1].plot(total_overlap, label=f'((cycle: {cycle}): {total_overlap[-1]:.4f}', linestyle='-', linewidth=1)

        if num == 4:
             break

    for energy in np.unique(w):
        if energy != e_ground:
            ax[0].axhline(y = energy, color='k', linestyle='--', linewidth=1.5)

    ax[0].axhline(y=e_ground, label=f'Exact E_g: {e_ground:.4f}', color='k', linestyle='--', linewidth=2)
    ax[0].set_xlabel('Iteration')
    ax[0].set_title('Cost')
    ax[0].legend()
    ax[1].axhline(y=1, label='Maximum overlap', color='k', linestyle='--', linewidth=2)
    ax[1].set_xlabel('Iteration')
    ax[1].set_title(f'Overlap (deg: {degeneration})')
    ax[1].legend()

    fig.suptitle(f'{title}', fontsize=16, fontweight = 'bold')
    fig.text(0.85, 0.98, f'qubits: {n_qubits}, # iter: {len(glob_history)}', ha='right', fontsize=12, fontstyle='italic');
    fig.text(0.1, 0.98, f'ansatz: {ansatz}', ha='left', fontsize=12, fontstyle='italic');
    fig.text(0.1, 0.93, f'runtime: {horas:.0f}:{minutos:.0f}:{segundos:.1f}', ha='left', fontsize=12, fontstyle='italic');
    fig.text(0.5, 0.9, f'optimizer: {optimizer}', ha='center', fontsize=12, fontstyle='italic');
    
    return fig, ax

def diagonalizar(matrix, printing: bool = False):

    w, v = np.linalg.eig(matrix)

    e_ground = np.min(w)

    # ground_states = v[:, np.where(w == np.min(w))[0]]

    ground_states = v[:, np.where(np.isclose(w, e_ground))[0]]
    degeneration = np.sum(np.isclose(w, e_ground))

    if printing:
        print(f'La matriz del Hamiltoniano es:\n {matrix}\n\n'
            'Este hamiltoniano tiene los siguientes estados propios con sus correspondientes energías (calculadas de manera exacta):\n')

        for i in range(len(w)):
            print(f'Autovalor {i+1}: {w[i]}')
            print(f'Autovector {i+1}: {v[:, i]}\n')
            
        print(f'El ground state (exacto) asociado al Hamiltoniano del problema es: \n {ground_states} \n'
            f'con una energía de: {e_ground} y degeneración: {degeneration}')
    
    return e_ground, (w,v), degeneration
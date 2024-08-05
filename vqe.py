import sys
import os
from typing import Callable, Literal, Dict, Any, Tuple, List, Iterable
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal
#from qiskit.utils import algorithm_globals
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
from qiskit.visualization import plot_state_hinton 
from modules_v1 import ansätze, hamiltonians


def construct_matrix_ising(n_qubits: int, J1: tuple = (1,), scope: int = 1, J2: tuple = (None, None, None)):
    
    if len(J1) != scope:
        raise ValueError("La longitud de J1 debe coincidir con el valor de alcance.")
    
    I = np.array([[1, 0], [0,  1]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1,  0]])
    Y = np.array([[0, -1j], [1j,  0]])

    interaction_terms = []
    
    for i in range(1,scope+1):
        interaction_term = Z
        for j in range(i):
            if j == i-1:
                interaction_term = np.kron(interaction_term,Z)
            else:
                interaction_term = np.kron(interaction_term,I)
        interaction_terms.append(interaction_term)


    matrices = (X,Y,Z)

    terms = []

    if J1 and J1[0] != 0 :
        try:
            for alcance, interact_term in enumerate(interaction_terms, start = 1):
                for i in range(n_qubits-alcance):
                    term = interact_term
                    for j in range(i): # Identidades por la izquierda
                        term = np.kron(I,term)
                    for j in range(n_qubits-alcance-1-i): # Identidades por la derecha (restamos 2 porque son dos matrices Z) range(n_qubits-alcance-1-i) ahora restamos alcance+1
                        term = np.kron(term,I)
                    terms.append(J1[alcance - 1] * term * -1)
        except Exception as e:
            print(e)

    for mat, coeff in zip(matrices, J2):
        if coeff is not None and coeff != 0:
            try:
                for i in range(n_qubits):
                    term = mat
                    for j in range(i):
                        term = np.kron(I,term)
                    for j in range(n_qubits-i-1):
                        term = np.kron(term,I)
                    terms.append(coeff * term * -1)
            except Exception as e:
                print(e)
                sys.exit(1)
            
    return sum(terms)


def change_basis(h_term):

    change_basis = ()
    
    # print(f'El término que quiero medir es: {h_term}')
    for qubit, operator in h_term:
        if operator in ('I', 'Z'):
            change_basis += ((qubit, 'I'),)
        elif operator == 'X':
            change_basis += ((qubit, 'H'),)
        elif operator == 'Y':
            change_basis += ((qubit, 'S'), ) # Pongo S para conservar 1 sólo char, la otra función lo tendrá en cuenta para hacer HS^dag
        else:
            print(f'He encontrado un char ({operator}) en un término del hamiltoniano que no entiendo')
    
    return change_basis



def prepare_to_measure(qc: QuantumCircuit, basis_term = None):
    
    n_qubits = qc.num_qubits
    i_qubit = 0
    if basis_term is not None:

        for qubit, operator in basis_term:
            if i_qubit < qubit:
                for i in range(i_qubit, qubit):
                    qc.id(i)
            if operator == 'I':
                qc.id(qubit)
            elif operator == 'H':
                qc.h(qubit)
            elif operator == 'S':
                qc.sdg(qubit)
                qc.h(qubit)

            i_qubit = qubit +1
        for i in range(i_qubit, n_qubits):
            qc.id(i)

    qc.measure(range(n_qubits), range(n_qubits))
    
    return qc


def computar_circuito(circuit, num_shots = 1024):
    
    sim = qiskit_aer.AerSimulator()
    job = sim.run(circuit, shots = num_shots, memory = True)
    result = job.result()
    counts_MSBF = result.get_counts()
    probs = {}
    
    for key in counts_MSBF:
        nueva_key = key[::-1]
        probs[nueva_key] = counts_MSBF[key]/num_shots
    
    return result, probs

def valor_esperado_Pauli_string(estado, pauli_string):

    e = 1

    for qubit, _ in pauli_string:
        if estado[qubit] == '0':
            e *= 1
        else:
            e *= -1

    # print(f'El observable {pauli_string} tiene un valor esperado en el estado {estado} de {e}')

    return e

# Los parámetros a optimizar deben ser el primer argumento
class ExpectedValueHIsing1DWrapper:

    def __init__(self) -> None:
        self.value = None
        self.state = None
    
    def __call__(self, theta_values: Iterable, h_terms: dict, ansatz: QuantumCircuit, magn: bool = False) -> float:
        # Ahora h_terms va a ser en minúscula y va a ser un diccionario:  {((qubit, 'operador'), (qubit, 'operador')): coef}
        self.value = 0
        # print(f'\nVeamos el valor esperado del Hamiltoniano: {h_terms}')

        mag_per_state: dict = {}
        
        for term_tuples, coef in h_terms.items():
            
            basis_term = change_basis(h_term = term_tuples)
            
            qc = ansatz.assign_parameters(theta_values, inplace = False)

            qc_measure = prepare_to_measure(qc = qc, basis_term = basis_term)

            result, probs = computar_circuito(circuit = qc_measure)

            self.state = result.get_statevector()

            e = 0

            if not magn:

                for state in probs:

                    valor = valor_esperado_Pauli_string(estado = state, pauli_string = term_tuples)
                    e += valor * probs[state] * coef

                self.value += e
            
            else:
                
                for state in probs:
                    if state not in mag_per_state:
                        mag_per_state[state] = 0
                    valor = valor_esperado_Pauli_string(estado = state, pauli_string = term_tuples)
                    mag_per_state[state] += valor * probs[state] * coef
            
        if magn:

            for state in mag_per_state:

                self.value += np.abs(mag_per_state[state])


    
        # print(f'En esta iteración el Hamiltoniano tiene un valor esperado de: {self.value}, probs: {probs}\n')

        return self.value


def vqe(n_qubits: int, hamiltoniano: dict, num_iterations: int = 1, ansatz_name: ansätze.AnsatzName = None, theta_values: Iterable = None, **ansatz_kwargs: Any):
    start = time.perf_counter()
    glob_runtime: float = 0.0


    if ansatz_name is not None:
        ansatz_fun = ansätze.ansatz_functions[ansatz_name]
        ansatz = ansatz_fun(n_qubits = n_qubits, **ansatz_kwargs)
    else:
        print("Es necesario expecificar un tipo de ansatz")
        sys.exit(1)
    
    expected_value_H_ising_1D = ExpectedValueHIsing1DWrapper()
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
        result, probs = computar_circuito(circuit=final_circuit)
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
                   f'm_z: {magnet_avg:.2f}', linestyle='-')

        # Pintar la segunda curva en el segundo subplot
        # for i in overlap:
        
        ax[1].plot(total_overlap, label=f'((cycle: {cycle}): {total_overlap[-1]:.4f}', linestyle='-')

        if num == 4:
             break


    ax[0].axhline(y=e_ground, label=f'Exact E_g: {e_ground:.4f}', color='k', linestyle='--')
    ax[0].set_xlabel('Iteration')
    ax[0].set_title('Cost')
    ax[0].legend()
    ax[1].axhline(y=1, label='Maximum overlap', color='k', linestyle='--')
    ax[1].set_xlabel('Iteration')
    ax[1].set_title(f'Overlap (deg: {degeneration})')
    ax[1].legend()

    fig.suptitle(f'{title}', fontsize=16, fontweight = 'bold')
    fig.text(0.85, 0.98, f'qubits: {n_qubits}', ha='right', fontsize=12, fontstyle='italic');
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
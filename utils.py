import numpy as np
import sys
from typing import Iterable
import qiskit
from qiskit import QuantumCircuit
import qiskit_aer



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


def computar_circuito(circuit, num_shots = 1024, noise_model = None):
    
    sim = qiskit_aer.AerSimulator(noise_model = noise_model)
    circuit_transpiled = qiskit.transpile(circuit, sim)
    job = sim.run(circuit_transpiled, shots = num_shots, memory = True)
    result = job.result()
    counts_LSBF = result.get_counts() 
    probs = {}
    
    for key in counts_LSBF:
        nueva_key = key[::-1]
        probs[nueva_key] = counts_LSBF[key]/num_shots
    
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

def separate_terms(hamiltonian: dict):
    same_compute: dict = {}
    different_comput: dict = {}
    for key, value in hamiltonian.items():

        all_z_or_i = True

        for _, gate in key:
            if gate not in ('Z', 'I'):
                all_z_or_i = False
        
        if all_z_or_i:
            same_compute[key] = value
        else:
            different_comput[key] = value
    return same_compute, different_comput

# Los parámetros a optimizar deben ser el primer argumento
class ExpectedValueHIsing1DWrapper:

    def __init__(self, noise_model) -> None:
        self.value = None
        self.state = None
        self.noise_model = noise_model
    
    def __call__(self, theta_values: Iterable, h_terms: dict, ansatz: QuantumCircuit, magn: bool = False) -> float:
        # Ahora h_terms va a ser en minúscula y va a ser un diccionario:  {((qubit, 'operador'), (qubit, 'operador')): coef}
        self.value = 0
        # print(f'\nVeamos el valor esperado del Hamiltoniano: {h_terms}')

        same_compute_terms, different_compute_terms = separate_terms(hamiltonian = h_terms)
        

        qc_same_compute = ansatz.assign_parameters(theta_values, inplace = False)
        qc_same_compute.measure(range(qc_same_compute.num_qubits), range(qc_same_compute.num_qubits))
        result, probs = computar_circuito(circuit = qc_same_compute, noise_model = self.noise_model)
        self.state = result.get_statevector()
        for state in probs:
            e = 0
            for term_tuples, coef in same_compute_terms.items():
                valor = valor_esperado_Pauli_string(estado = state, pauli_string = term_tuples)
                e += valor * probs[state] * coef
            if magn:
                self.value += np.abs(e)
            else:
                self.value += e


        for term_tuples, coef in different_compute_terms.items():
            basis_term = change_basis(h_term = term_tuples)
            
            qc = ansatz.assign_parameters(theta_values, inplace = False)

            qc_measure = prepare_to_measure(qc = qc, basis_term = basis_term)

            result, probs = computar_circuito(circuit = qc_measure, noise_model = self.noise_model)

            self.state = result.get_statevector()

            e = 0

            for state in probs:

                valor = valor_esperado_Pauli_string(estado = state, pauli_string = term_tuples)
                e += valor * probs[state] * coef

            self.value += e
            
            # else:
                
            #     for state in probs:
            #         if state not in mag_per_state:
            #             mag_per_state[state] = 0
            #         valor = valor_esperado_Pauli_string(estado = state, pauli_string = term_tuples)
            #         mag_per_state[state] += valor * probs[state] * coef
            
        # if magn:

        #     for state in mag_per_state:

        #         self.value += np.abs(mag_per_state[state])


    
        # print(f'En esta iteración el Hamiltoniano tiene un valor esperado de: {self.value}, probs: {probs}\n')

        return self.value
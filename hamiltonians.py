import sys
import os
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

def term_tuples_to_data(term_tuples: dict, n_qubits: int):

    I = np.array([[1, 0], [0,  1]])
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1,  0]])
    Y = np.array([[0, -1j], [1j,  0]])
    pauli_matrices = {
        'I': I,
        'X': X,
        'Y': Y,
        'Z': Z
    }

    terms = []

    for term, value in term_tuples.items():
        
        if not term:
            # Si el término es una tupla vacía, es un término de identidad
            term_matrix = np.identity(2 ** n_qubits) * value
            terms.append(term_matrix)
            continue
        qubit, op = term[0]

        for i in range(n_qubits):
            if i == 0:
                # Iniciar la matriz del término
                if qubit != 0:
                    term_matrix = I
                else:
                    term_matrix = pauli_matrices[op]
                    term = term[1:]  # Nos desprendemos del término
                    if term:
                        qubit, op = term[0]
                    else:
                        qubit, op = None, 'I'

            elif term and qubit == i:
                term_matrix = np.kron(term_matrix, pauli_matrices[op])
                term = term[1:]  # Nos desprendemos del término
                if term:
                    qubit, op = term[0]
                else:
                    qubit, op = None, 'I'
            else:
                term_matrix = np.kron(term_matrix, I)
        
        terms.append(term_matrix * value)
    
    return sum(terms) 

        # for qubit, operator in term:

        #     for _ in range(qubit-i_qubit-1):
        #         term_matrix = np.kron(term_matrix,I)
        #         print(f'Le meto identidad y la matriz es: \n{term_matrix}, con una forma de {np.shape(term_matrix)}')
        #     term_matrix = np.kron(term_matrix, pauli_matrices[operator])
        #     print(f'Le meto el operador {operator} y la matriz es: \n{term_matrix}, con una forma de {np.shape(term_matrix)}')
        #     i_qubit = qubit + 1
            
        # print(f'i_qubit vale: {i_qubit}')
        # for _ in range(n_qubits - i_qubit -1):
        #     term_matrix = np.kron(term_matrix,I)
        # terms.append(term_matrix)
    


def ising_Hamiltonian_1D(n_qubits: int, J1: tuple = (1,), scope: int = 1, J2: tuple = (None, None, None)):

    """
    Genera el Hamiltoniano de Ising 1D para una cadena de espines.

    Args:
        n_qubits (int): Número de qubits en la cadena.
        J1 (tuple, optional): Coeficientes para términos de interacción Z-Z, por alcance. Por defecto es (1,).
        scope (int, optional): Alcance máximo de interacción entre espines. Por defecto es 1.
        J2 (tuple, optional): Coeficientes para términos de Pauli X, Y, Z. Por defecto es (None, None, None).

    Returns:
        dict: Diccionario con términos del Hamiltoniano representados por tuplas de tuplas.

    Raises:
        ValueError: Si los tamaños de `J1` y `scope` no coinciden, o si `J1` tiene longitud incorrecta.
        
    """

    if scope == 0:
        if len(J1) != 1:
            raise ValueError("Especifice una única constante J1.")
    elif len(J1) != scope:
        raise ValueError("La longitud de J1 debe coincidir con el valor de alcance.")
    
    matrices = ('X', 'Y', 'Z')
    h_terms = {}
    
    def add_pauli_term(dictionary: dict, key: tuple, value: complex):
        if key in dictionary:
            dictionary[key] += value
        else:
            dictionary[key] = value

    for alcance, coeff in enumerate(J1, start = 1):
        if coeff is not None and coeff != 0:
            if scope == 0:
                # Magnetización
                for qubit in range(n_qubits):
                    add_pauli_term(dictionary = h_terms, key = ((qubit, 'Z'),),value = J1[0] * -1) # * -1 porque el hamiltoniano de Ising esta definido como -J1
            else:
                for qubit in range(n_qubits-alcance):
                    add_pauli_term(dictionary = h_terms, key = ((qubit, 'Z'), (qubit + alcance, 'Z')), value = coeff * -1)


    for mat, coeff in zip(matrices, J2):
        if coeff is not None and coeff != 0:
            for qubit in range(n_qubits):
                add_pauli_term(dictionary = h_terms, key = ((qubit, mat),), value = coeff * -1)

    return h_terms


# def ising_Hamiltonian_1D(n_qubits: int, J1: tuple = (-1,), scope: int = 1, J2: tuple =(None, None, None)):

#     if scope == 0:
#         if len(J1) != 1:
#             raise ValueError("Especifice una única constante J1.")
#     elif len(J1) != scope:
#         raise ValueError("La longitud de J1 debe coincidir con el valor de alcance.")

#     terminos_acoplo = []
#     terminos_campo = [] # CAMPO TRANSVERSAL!!!!

#     matrices = ('X', 'Y', 'Z')

#     for alcance, coeff in enumerate(J1, start = 1):
#         if coeff != 0:
#             try:
#                 for i in range(n_qubits-alcance):
#                     operators = ["I"] * n_qubits
#                     operators[i] = "Z"
#                     if scope == 0:
#                         pass
#                     else:
#                         operators[i + alcance] = "Z"
#                     cadena = ''.join(operators)
#                     terminos_acoplo.append((coeff,cadena))
#                 if scope == 0:
#                     operators = ["I"] * n_qubits
#                     operators[n_qubits-1] = "Z"
#                     cadena = ''.join(operators)
#                     terminos_acoplo.append((coeff, cadena))
#             except Exception as e:
#                 print(e)
#                 sys.exit(1)
        

#     for mat, coeff in zip(matrices, J2):
#         if coeff is not None and coeff != 0:
#             try:
#                 for i in range(n_qubits):
#                     operators = ["I"] * n_qubits
#                     operators[i] = mat
#                     cadena = ''.join(operators)
#                     terminos_campo.append((coeff, cadena))
#             except Exception as e:
#                 print(e)
#                 sys.exit(1)
        

#     print(f'Términos de acoplo: {terminos_acoplo}')
#     print(f'Términos de campo externo: {terminos_campo}')

#     H_terms = terminos_acoplo + terminos_campo
#     # acoplo = SparsePauliOp(terminos_acoplo,coeffs=[1.0 for i in range(len(terminos_acoplo))])
#     # campo = SparsePauliOp(terminos_campo,coeffs=[1.0 for i in range(len(terminos_campo))])
#     # Ham = acoplo + campo
#     # np.set_printoptions(precision=2, suppress=True, linewidth=100)
#     # np.set_printoptions(threshold=sys.maxsize)
#     # np.set_printoptions(linewidth=np.inf)
#     # default_options = np.get_printoptions()
#     # print(f'La matriz de la parte de acoplo que proporciona Qiskit es:\n {acoplo.to_matrix().real}')
#     # print(f'La matriz de la parte de campo externo que proporciona Qiskit es:\n {campo.to_matrix().real}')
#     # print(f'La matriz del Hamiltoniano total que proporciona Qiskit es:\n {Ham.to_matrix().real}')
#     # np.set_printoptions(**default_options)

#     return H_terms
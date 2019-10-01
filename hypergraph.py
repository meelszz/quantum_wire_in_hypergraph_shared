import numpy as np
from numpy import linalg as LA
import itertools
import traceback
import logging

# utility
x_matrix = np.array([[0, 1],
                     [1, 0]])

i_matrix = np.array([[1, 0],
                     [0, 1]])

ones_matrix = np.array([[0, 0],
                        [0, 1]])

z_matrix = np.array([[1, 0],
                     [0, -1]])

h_matrix = (np.sqrt(.5))*np.array([[1,1],
                                       [1,-1]])

plus_state = np.array([[np.sqrt(.5)],
                       [np.sqrt(.5)]])


# construct hadamard matrix on a state of size num_v on qubit q
def construct_hadamard(num_v, q):
    if q == 0:
        out = h_matrix
    else:
        out = i_matrix
    for i in range(1, num_v):
        if i == q:
            out = np.kron(out, h_matrix)
        else:
            out = np.kron(out, i_matrix)
    return out


# Create n qubit identity matrix : O(n)
def initialize_i(num_v):
    i = i_matrix
    for n in range(num_v - 1):
        i = np.kron(i, i_matrix)
    return i


# Initialize state as |+>_n : O(n)
def initialize_state(num_v):
    state = plus_state
    for n in range(num_v-1):
        state = np.kron(state, plus_state)
    return state


# Create matrices Z_1 to Z_n : O(n^2)
def initialize_loz(num_v):
    loz = [0] * num_v
    loz[0] = z_matrix
    for k in range(1, num_v):
        loz[0] = np.kron(loz[0], i_matrix)
    for n in range(1, num_v):
        loz[n] = i_matrix
        for k in range(1, num_v):
            if n == k:
                loz[n] = np.kron(loz[n], z_matrix)
            else:
                loz[n] = np.kron(loz[n], i_matrix)
    return loz


# Create matrices X_1 to X_n : O(n^2)
def create_lox(num_v):
    lox = [0] * num_v
    lox[0] = x_matrix
    for k in range(1, num_v):
        lox[0] = np.kron(lox[0], i_matrix)
    for n in range(1, num_v):
        lox[n] = i_matrix
        for k in range(1, num_v):
            if n == k:
                lox[n] = np.kron(lox[n], x_matrix)
            else:
                lox[n] = np.kron(lox[n], i_matrix)
    return lox


# Create measurement operators
def create_m_ops(num_v):
    set_p = {}
    for i in range(1, num_v-1):
        if (num_v-i)%2 == 0:
            for case in list(itertools.product([0, 1], repeat=int(i))):
                # create unique key for case
                key = gen_case_key(case)
                # create proj matrix for measurement case
                p = i_matrix
                if not (num_v-i)/2 == int((num_v-i)/2):
                    print("halp")
                for n in range(1, int((num_v-i)/2)):
                    p = np.kron(p, i_matrix)
                for c in case:
                    if c == 1:
                        p = np.kron(p, (i_matrix - x_matrix) / 2)
                    else:
                        p = np.kron(p, (i_matrix + x_matrix) / 2)
                for n in range(0, int((num_v - i) / 2)):
                    p = np.kron(p, i_matrix)
                # update proj matrix set for case
                set_p[key] = p
    return set_p


# Create extraction matrices
# extr_ms[(n-1)th] is the nth extraction matrix assuming measurement result 0
# extr_ms[2*(n-1)th] is the nth extraction matrix assuming measurement result 1
def create_extr_matrices(num_v):
    extr_ms = [0] * (num_v - 2) * 2
    for a in range(num_v - 2):
        id = np.identity((int(2 ** (num_v - a) / 4)))
        dim = int(2 ** (num_v - a))
        dim2 = int((1 / 2) * (2 ** (num_v - a)))
        dim3 = int((3 / 4) * (2 ** (num_v - a)))
        dim4 = int((1 / 4) * (2 ** (num_v - a)))
        extr_ms[a] = np.zeros([dim2, dim])
        extr_ms[a][0:dim4, 0:dim4] = id
        extr_ms[a][dim4:dim2, dim2:dim3] = id
        index = (num_v - 2 + a)
        extr_ms[index] = np.zeros([dim2, dim])
        extr_ms[index][0:dim4, dim4:dim2] = id
        extr_ms[index][dim4:dim2, dim3:dim] = id
    return extr_ms


# Create Hadamard matrices used with extraction matrices - on qubit 1 with decreasing # of qubits
def create_loh(num_v):
    loh = [0] * (num_v - 2)
    for i in range(num_v-2):
        loh[i] = construct_hadamard(num_v-i, 1)
    return loh


# Generate unique key based on case
def gen_case_key(case):
    primes = (5, 7, 11, 13, 17, 19, 23, 29, 31)
    key = 0
    for i in range(len(case)):
        key = key + ((case[i]+1) * primes[i])
    return key

# create matrix to extract qubit q with measurement r from a num_v qubit state
def create_extr_matrix(num_v, q, r):
    E = np.zeros((2**(num_v-1), 2**num_v))
    chunk_size = int((2 ** num_v) / (2**q))
    I_chunk = np.identity(chunk_size)
    num_iterations = (2 ** num_v) / (chunk_size * 2)
    i = 0
    while i < num_iterations:
        if r == 0:
            E[i * chunk_size: i * chunk_size + chunk_size,
            2 * i * chunk_size: 2 * i * chunk_size + chunk_size] = I_chunk
        elif r == 1:
            E[i * chunk_size: i * chunk_size + chunk_size,
            2*i*chunk_size + chunk_size: 2*i*chunk_size + chunk_size + chunk_size] = I_chunk
        i += 1
    h = construct_hadamard(num_v, q-1)
    return E.dot(h)


class Hypergraph:
    def __init__(self, num_v):
        self.num_v = num_v
        self.loz = initialize_loz(num_v)
        self.i = initialize_i(num_v)
        self.m_ops = create_m_ops(num_v)
        self.extr_dict = {}
        self.loh = create_loh(num_v)
        self.locz = np.zeros((num_v, num_v, num_v, 2**num_v, 2**num_v))
        self.locz_order_4 = np.zeros((num_v, num_v, num_v, num_v, 2**num_v, 2**num_v))
        self.cz_multiplier = np.identity(2 ** num_v)
        self.h_state = 0

    # Construct state from CZ matrices : O(n) for small cases
    def construct_state(self, faces):
        self.cz_multiplier = np.identity(2 ** self.num_v)
        self.h_state = initialize_state(self.num_v)
        for f in faces:
            if np.count_nonzero(self.locz[f[0]][f[1]][f[2]]) == 0:
                cz = self.i - 2 * ((self.i - self.loz[f[0]]) / 2).dot((self.i - self.loz[f[1]]) / 2).dot(((self.i - self.loz[f[2]]) / 2))
                self.locz[f[0]][f[1]][f[2]] = cz
                self.cz_multiplier = self.cz_multiplier.dot(cz)
            else:
                self.cz_multiplier = self.cz_multiplier.dot(self.locz[f[0]][f[1]][f[2]])
        self.h_state = self.cz_multiplier.dot(self.h_state)

    # Construct state from CZ matrices : O(n) for small cases
    def construct_state_order_4(self, faces):
        self.cz_multiplier = np.identity(2 ** self.num_v)
        self.h_state = initialize_state(self.num_v)
        for f in faces:
            if np.count_nonzero(self.locz_order_4[f[0]][f[1]][f[2]][f[3]]) == 0:
                cz = self.i - 2 * ((self.i - self.loz[f[0]]) / 2).dot((self.i - self.loz[f[1]]) / 2).dot(
                    ((self.i - self.loz[f[2]]) / 2)).dot(((self.i - self.loz[f[3]]) / 2))
                self.locz_order_4[f[0]][f[1]][f[2]][f[3]] = cz
                self.cz_multiplier = self.cz_multiplier.dot(cz)
            else:
                self.cz_multiplier = self.cz_multiplier.dot(self.locz[f[0]][f[1]][f[2]])
        self.h_state = self.cz_multiplier.dot(self.h_state)

    def normalize(self):
        if LA.norm(self.h_state) != 0:
            self.h_state = self.h_state / LA.norm(self.h_state)

    # Perform measurement on state : O(n)
    def perform_measurement(self, case):
        key = gen_case_key(case)
        p = self.m_ops[key]
        self.h_state = p.dot(self.h_state)
        self.normalize()
        return p

    def extract_bits(self, m, case):
        for i in range(len(m)):
            n = self.num_v
            key = gen_case_key((n, m[i], case[i]+1))
            if key not in self.extr_dict:
                self.extr_dict[key] = create_extr_matrix(n, m[i]-i+1, case[i])
            E = self.extr_dict[key]
            self.h_state = E.dot(self.h_state)
            self.num_v = n - 1

def all_w_degenerate(low):
    w_set = {}
    w_set[2] = 1
    for w in low:
        w = round(w, 6)
        if w in w_set:
            w_set[w] = 1
        else:
            w_set[w] = 0
    output = True
    for w in w_set:
        if w_set[w] == 0:
            output = False
    return output

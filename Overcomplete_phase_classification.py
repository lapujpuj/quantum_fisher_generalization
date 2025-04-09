import numpy as np

import qutip as qt
import scipy
import scipy.io
import operator

import matplotlib.pyplot as plt
from functools import reduce
from Hamilotonian_phase_data import * 
from Training_utils import *





##parameters of program

n_qubits=3 #3 #3 ##number of qubits
depth=20 ##depth of quantum circuit

n_test=100 ##number test states sampled from distribution
maxiter_opt= 500 #-1#100000, set to zero to only compute gradients, -1: do nothing
model=0 ##0: hardware efficient circuit, 1: XY ansatz
type_state=1#input states for training, 0: haar random, 1: product states 2: 1 particle states
type_state_test=type_state ##input states for test, 0: haar random, 1: product states 2: 1 particle states
n_particles=1 ##for type_state==2


depth_input=20 ##additional depth added for preparing particle conserved states
fix_depth=20 ##add unitary of fixed depth at end of circuit to make more random



n_training_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] #number of training states to be used for training
n_test = 10
depth = 20
n_qubits = 3

print("Number training states",n_training_list)


## Create training and test states

print(" == == Generating training and test states ===")
(training_data_dataset, training_data_feature_dataset), training_target_dataset = generate_samples(n_qubits, np.max(n_training_list))
(test_states_dataset, test_states_feature_dataset) , test_target_dataset = generate_samples(n_qubits, n_test)


def cost(x, in_unitary=[]):
    ##compute cost function for optimization
    global callback_training,callback_test
    if(type(in_unitary)==list):
        encode_unitary=get_unitary(n_qubits,depth,x,add_fixed_unitary=add_fixed_unitary,ini_state=opId,model=model,opEntangler=opEntangler)
    else:
        encode_unitary=in_unitary
        
    ##get train error
    cost_train_states=[]
    for k in range(n_training):
        output_state = encode_unitary * training_states[k]
        # Measure the expectation value of the first qubit's Z operator
        first_qubit_z = qt.sigmaz()
        measured_value = qt.expect(first_qubit_z, output_state.ptrace(0))
        loss = (measured_value - training_target[k]) ** 2
        cost_train_states.append(loss)
        
    ##get test error
    cost_test_states=[]
    for k in range(n_test):
        output_state = encode_unitary * test_states[k]
        # Measure the expectation value of the first qubit's Z operator
        first_qubit_z = qt.sigmaz()
        measured_value = qt.expect(first_qubit_z, output_state.ptrace(0))
        loss = (measured_value - test_target[k]) ** 2
        cost_test_states.append(loss)
    ##these variables are given to callback function to be recorded
    callback_training=1-np.mean(cost_train_states)
    callback_test=1-np.mean(cost_test_states)
    return callback_training



def cost_grad(x):
    ##get gradient for optimisation function
    delta_x=1.49011612e-08 ##same as used by LBFGS by default
    grad_list=np.zeros(len(x))
    grad_unitary_list,circuit_orig=get_unitary_grad(n_qubits,depth,x,add_fixed_unitary=add_fixed_unitary,ini_state=opId,model=model,opEntangler=opEntangler,delta_x=delta_x)
    ##compute train error
    cost_train=[]
    n_training=len(training_states)
    for k in range(n_training):
        output_state = circuit_orig * training_states[k]
        # Measure the expectation value of the first qubit's Z operator
        first_qubit_z = qt.sigmaz()
        measured_value = qt.expect(first_qubit_z, output_state.ptrace(0))
        loss = (measured_value - training_target[k]) ** 2
        cost_train.append(loss)
    res=1-np.mean(cost_train)
    
    ##compute gradient of train error
    for q in range(len(x)):
        cost_shift=[]
        encode_unitary=grad_unitary_list[q]
        for k in range(n_training):
            output_state = encode_unitary * training_states[k]
            # Measure the expectation value of the first qubit's Z operator
            first_qubit_z = qt.sigmaz()
            measured_value = qt.expect(first_qubit_z, output_state.ptrace(0))
            loss = (measured_value - training_target[k]) ** 2
            cost_shift.append(loss)
        ##use finite difference method to get gradient
        grad_list[q]=(np.mean(cost_shift)-res)/delta_x
    return grad_list

      

def callback_opt(x):
    #is called after every iteration of scipy minimize
    ##saves training and test cost
    global callback_training,callback_test,cost_training_list,cost_test_list
    cost_training_list.append(callback_training)
    cost_test_list.append(callback_test)


cost_train_final_list = []
cost_test_final_list = []
rank_DQFIM_list = []


noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
cost_train_final_lists = {noise_ratio: [] for noise_ratio in noise_levels}
cost_test_final_lists = {noise_ratio: [] for noise_ratio in noise_levels}
rank_DQFIM_lists = {noise_ratio: [] for noise_ratio in noise_levels}


for noise_ratio in noise_levels:
    print(f"=== Processing with noise ratio: {int(noise_ratio * 100)}% ===")

    cost_train_final_list = []
    cost_test_final_list = []
    rank_DQFIM_list = []

    for p in range(len(n_training_list)):
        n_training = n_training_list[p]
        print("Number training states", n_training)

        levels = 2
        opZ = [genFockOp(qt.sigmaz(), i, n_qubits, levels) for i in range(n_qubits)]
        opX = [genFockOp(qt.sigmax(), i, n_qubits, levels) for i isn range(n_qubits)]
        opY = [genFockOp(qt.sigmay(), i, n_qubits, levels) for i in range(n_qubits)]
        opId = genFockOp(qt.qeye(levels), 0, n_qubits)

        if model == 0:
            if n_qubits == 2:
                entangling_gate_index = [[0, 1]]
            else:
                entangling_gate_index = [[2 * j, (2 * j + 1) % n_qubits] for j in range(int(np.ceil(n_qubits / 2)))] + \
                                        [[2 * j + 1, (2 * j + 2) % n_qubits] for j in range((n_qubits) // 2)]
        elif model == 1:
            if n_qubits == 2:
                entangling_gate_index = [[0, 1]]
                entangling_gate_index2 = [[0, 1]]
            else:
                entangling_gate_index = [[2 * j, (2 * j + 1) % n_qubits] for j in range(int(np.ceil(n_qubits / 2)))]
                entangling_gate_index2 = [[2 * j + 1, (2 * j + 2) % n_qubits] for j in range((n_qubits) // 2)]

        if model == 0:
            if n_qubits > 1:
                opEntangler = [prod([qt.qip.operations.cnot(n_qubits, j, k) for j, k in entangling_gate_index[::-1]])]
            else:
                opEntangler = [opId]
        elif model == 1:
            if n_qubits > 1:
                opEntangler = [prod([qt.qip.operations.sqrtiswap(n_qubits, [j, k]) for j, k in entangling_gate_index[::-1]]),
                                prod([qt.qip.operations.sqrtiswap(n_qubits, [j, k]) for j, k in entangling_gate_index2[::-1]])]
            else:
                opEntangler = [opId]

        dims_mat = [[2] * n_qubits, [2] * n_qubits]
        dims_vec = [[2] * n_qubits, [1] * n_qubits]

        def get_circuit_parameters(depth, n_qubits, model):
            if model == 0:
                n_circuit_parameters = 2 * depth * n_qubits
            elif model == 1:
                n_circuit_parameters = depth * (n_qubits // 2)
            return n_circuit_parameters

        n_circuit_parameters = get_circuit_parameters(depth, n_qubits, model)
        print("Number parameters", n_circuit_parameters)

        if fix_depth > 0:
            fix_depth_n_parameters = get_circuit_parameters(fix_depth, n_qubits, model)
            if model in [2, 3]:
                fix_circuit_params = rng.integers(0, 4, fix_depth_n_parameters) * np.pi / 2
            else:
                fix_circuit_params = rng.random(fix_depth_n_parameters) * 2 * np.pi
            add_fixed_unitary = get_unitary(n_qubits, fix_depth, fix_circuit_params, add_fixed_unitary=[], ini_state=opId, model=model, opEntangler=opEntangler)
        else:
            add_fixed_unitary = []

        target_circuit_params = rng.random(n_circuit_parameters) * 2 * np.pi
        data_unitary = get_unitary(n_qubits, depth, target_circuit_params, add_fixed_unitary=add_fixed_unitary, ini_state=opId, model=model, opEntangler=opEntangler)

        training_states, training_target = training_data_dataset[:n_training], training_target_dataset[:n_training]
        test_states, test_target = test_states_dataset[:n_test], test_target_dataset[:n_test]

        training_target = inject_label_noise(training_target, noise_ratio)
        test_target = inject_label_noise(test_target, noise_ratio)

        options = {"maxiter": maxiter_opt}
        x0 = np.zeros(n_circuit_parameters)
        cost_training_list = []
        cost_test_list = []

        ini_val = cost(x0)
        callback_opt(x0)

        res_opt = scipy.optimize.minimize(cost, x0, jac=cost_grad, method="BFGS", options=options, callback=callback_opt)
        final_x = res_opt["x"]
        final_cost = res_opt["fun"]
        n_iterations = res_opt["nit"]

        encode_unitary = get_unitary(n_qubits, depth, final_x, add_fixed_unitary=add_fixed_unitary, ini_state=opId, model=model, opEntangler=opEntangler)

        cost_train_final = cost_training_list[-1]
        cost_test_final = cost_test_list[-1]

        print("Train cost", cost_train_final, "test cost", cost_test_final, "iterations", n_iterations)

        cost_train_final_list.append(cost_train_final)
        cost_test_final_list.append(cost_test_final)

        xr = rng.random(n_circuit_parameters) * 2 * np.pi
        qfi_training_dm_eigvals, qfi_training_dm = get_qfi_eigvals(xr,training_states,test_states,n_qubits,depth,add_fixed_unitary,opId,opEntangler,model)

        rank_DQFIM = np.sum(qfi_training_dm_eigvals > 10 ** -6)
        rank_DQFIM_list.append(rank_DQFIM)

        print("rank of data Quantum Fisher information metric", rank_DQFIM)

    rank_DQFIM_lists[noise_ratio] = rank_DQFIM_list
    cost_train_final_lists[noise_ratio] = cost_train_final_list
    cost_test_final_lists[noise_ratio] = cost_test_final_list


# Plot R_L vs L for different noise levels
plt.figure(figsize=(10, 6))
for noise_ratio, rank_DQFIM_list in rank_DQFIM_lists.items():
    plt.plot(n_training_list, rank_DQFIM_list, label=f"Noise: {int(noise_ratio * 100)}%")
plt.ylabel(r'$R_L$')
plt.xlabel("L")
plt.title("Rank of data DQFIM vs number of training states for different noise levels")
plt.legend(loc="upper right")
plt.savefig("figure/Rank_DQFIM_vs_training_states_noise_levels.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot train cost vs L for different noise levels
plt.figure(figsize=(10, 6))
for noise_ratio, cost_train_final_list in cost_train_final_lists.items():
    plt.plot(n_training_list, cost_train_final_list, label=f"Noise: {int(noise_ratio * 100)}%")
plt.ylabel('Train cost')
plt.xlabel("L")
plt.title("Train cost vs number of training states for different noise levels")
plt.legend(loc="upper right")
plt.savefig("figure/Train_cost_vs_training_states_noise_levels.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot test cost vs L for different noise levels
plt.figure(figsize=(10, 6))
for noise_ratio, cost_test_final_list in cost_test_final_lists.items():
    plt.plot(n_training_list, cost_test_final_list, label=f"Noise: {int(noise_ratio * 100)}%")
plt.ylabel('Test cost')
plt.xlabel("L")
plt.title("Test cost vs number of training states for different noise levels")
plt.legend(loc="upper right")
plt.savefig("figure/Test_cost_vs_training_states_noise_levels.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot generalization vs L for different noise levels
plt.figure(figsize=(10, 6))
for noise_ratio in cost_train_final_lists:
    ct = cost_train_final_lists[noise_ratio]
    ctest = cost_test_final_lists[noise_ratio]
    plt.plot(n_training_list, np.array(ct) - np.array(ctest), label=f"Noise: {int(noise_ratio * 100)}%")
plt.ylabel('Generalization')
plt.xlabel("L")
plt.title("Generalization vs number of training states for different noise levels")
plt.legend(loc="upper right")
plt.savefig("figure/Generalization_vs_training_states_noise_levels.png", dpi=300, bbox_inches='tight')
plt.show()





""" A parallele word where everything is possible and nothing is real."""

# import numpy as np
# from qutip import tensor, basis, sigmax, sigmaz, qeye

# def ising_hamiltonian(J, h, n_qubits=3):
#     """Build the transverse-field Ising Hamiltonian for 3 qubits."""
#     H = 0
#     for i in range(n_qubits):
#         Zi = tensor([sigmaz() if j == i else qeye(2) for j in range(n_qubits)])
#         Zj = tensor([sigmaz() if j == (i + 1) % n_qubits else qeye(2) for j in range(n_qubits)])
#         Xi = tensor([sigmax() if j == i else qeye(2) for j in range(n_qubits)])
#         H += -J * Zi * Zj - h * Xi
#     return H

# def classify_phase(J, h, epsilon=0.2):
#     """Classify the phase based on J and h."""
#     if J > h + epsilon:
#         return "ferromagnetic"
#     elif h > J + epsilon:
#         return "paramagnetic"
#     else:
#         return "critical"

# def get_input_state_ising():
#     """Sample J and h, return ground state and phase label."""
#     J = np.random.uniform(0.1, 2.0)
#     h = np.random.uniform(0.1, 2.0)
#     H = ising_hamiltonian(J, h, n_qubits=3)
#     eigvals, eigstates = H.eigenstates()
#     ground_state = eigstates[0]
#     phase_label = classify_phase(J, h)
#     return ground_state, phase_label

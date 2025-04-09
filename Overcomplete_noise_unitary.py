# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:36:54 2022

Program to learn unitaries
Study how ansatz and data affects training

Computes data Quantum Fisher information metric (DQFIM)
Rank of the DQFIM can be used to determine number of circuit parameters
and training data needed to converge to global minimum and generalization

Implements training with hardware efficient circuit or XY model with BFGS algorithm
Training states can be either Haar random, product states, or 
particle number conserved states


@author: Tobias Haug, Imperial College London
tobias.haug@u.nus.edu
"""


import numpy as np

import qutip as qt
import scipy
import scipy.io
import operator

import matplotlib.pyplot as plt
from functools import reduce


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


##go through different number of training states to
# if(model==0):
#     n_training_list=np.unique(np.array(np.round(10**np.linspace(0,np.log10(2*2**n_qubits),num=11)),dtype=int))
# if(model==1):
#     n_training_list=[1,2,3,5,8]

n_training_list=np.unique(np.array(np.round(10**np.linspace(0,np.log10(2*2**n_qubits),num=11)),dtype=int))
n_training_list=[1,2,5,10, 50, 100, 200]
print("Number training states",n_training_list)
                          

rng = np.random.default_rng()
datapoints=len(n_training_list)


# def inject_label_noise(states_target, noise_ratio):
#     """Inject noise into a fraction of the training labels."""
#     noisy_states = states_target.copy()
#     n_total = len(noisy_states)
#     n_noisy = int(noise_ratio * n_total)
#     idx_to_corrupt = rng.choice(n_total, n_noisy, replace=False)
#     for i in idx_to_corrupt:
#         # Replace the label with a randomly chosen one (different from current)
#         new_state = get_input_state(n_qubits, depth_input, model, type_state)
#         noisy_states[i] = new_state
#     return noisy_states

from qutip import  tensor


from qutip import Qobj, basis, qeye

def amplitude_damping_kraus(gamma):
    E0 = Qobj([[1, 0], [0, (1 - gamma) ** 0.5]])
    E1 = Qobj([[0, gamma ** 0.5], [0, 0]])
    return [E0, E1]

def phase_damping_kraus(gamma):
    E0 = Qobj([[1, 0], [0, (1 - gamma) ** 0.5]])
    E1 = Qobj([[0, 0], [0, gamma ** 0.5]])
    return [E0, E1]

def bit_flip_kraus(gamma):
    E0 = ((1 - gamma) ** 0.5) * qeye(2)
    E1 = (gamma ** 0.5) * Qobj([[0, 1], [1, 0]])  # Pauli-X
    return [E0, E1]


def apply_kraus_channel(rho, kraus_ops):
    return sum([E * rho * E.dag() for E in kraus_ops])



def apply_noise_to_qubit(rho, noise_type='amplitude', gamma=0.1):
    """Apply selected noise channel to a single-qubit density matrix."""
    if noise_type == 'amplitude':
        return apply_kraus_channel(rho, phase_damping_kraus(gamma))
    elif noise_type == 'dephasing':
        return apply_kraus_channel(rho, bit_flip_kraus(gamma))
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")


def inject_label_noise(states_target, gamma=0.05, noise_type='amplitude', n_qubits=3):
    """Apply controlled noise to all training targets."""
    noisy_states = []

    for state in states_target:
        noisy_qb_states = []
        for q in range(n_qubits):
            rho_q = state.ptrace(q)
            noisy_rho_q = apply_noise_to_qubit(rho_q, noise_type=noise_type, gamma=gamma)
            noisy_qb_states.append(noisy_rho_q)
        noisy_state = tensor(noisy_qb_states).unit()
        noisy_states.append(noisy_state)

    return noisy_states







def prod(factors):
    return reduce(operator.mul, factors, 1)

def flatten(l):
    return [item for sublist in l for item in sublist]

#tensors operators together 
def genFockOp(op,position,size,levels=2,opdim=0):
    opList=[qt.qeye(levels) for x in range(size-opdim)]
    opList[position]=op
    return qt.tensor(opList)


def get_unitary(n_qubits,depth,circuit_params,add_fixed_unitary,ini_state=[],model=0,opEntangler=[]):
    """
    computes unitary with given parameters and model
    """

    n_circuit_parameters=len(circuit_params)

    circuit_state=ini_state
    counter_params=0
    for i in range(depth):
        if(model==0):
            rot_op_list1=[]
            ##y rotations
            for j in range(n_qubits):
                param_index=i*n_qubits*2+j
                counter_params+=1
                angle=circuit_params[param_index]
                #rot_op=qt.qip.operations.ry(angle,n_qubits,j)

                rot_op=qt.qip.operations.ry(angle)

                rot_op_list1.append(rot_op)

                #circuit_state=rot_op*circuit_state
                
            ##z rotations
            circuit_state=qt.tensor(rot_op_list1)*circuit_state
            rot_op_list2=[]
            for j in range(n_qubits):
                param_index=i*n_qubits*2+n_qubits+j
                counter_params+=1
                angle=circuit_params[param_index]
                
                #rot_op=qt.qip.operations.rz(angle,n_qubits,j)
                rot_op=qt.qip.operations.rz(angle)

                rot_op_list2.append(rot_op)

                #circuit_state=rot_op*circuit_state

            circuit_state=qt.tensor(rot_op_list2)*circuit_state
                
            circuit_state=opEntangler[i%len(opEntangler)]*circuit_state

            
        elif(model==1): ##reduced swqrtiswap + z
            rot_op_list1=[]
            for j in range(n_qubits//2):
                param_index=i*(n_qubits//2)+j
                counter_params+=1
                angle=circuit_params[param_index]

                rot_op=qt.qip.operations.rz(angle)
                circuit_state=qt.qip.operations.gate_expand_1toN(rot_op, n_qubits, 2*j)*circuit_state


            circuit_state=opEntangler[i%len(opEntangler)]*circuit_state
            

                
    if(type(add_fixed_unitary)!=list):
        circuit_state=add_fixed_unitary*circuit_state

    if(counter_params!=n_circuit_parameters):
        raise NameError("Number parameters used does not match")

    return circuit_state


def get_unitary_grad(n_qubits,depth,circuit_params,add_fixed_unitary,ini_state=[],model=0,opEntangler=[],delta_x=1.49011612e-08):
    ##compute unitaries needed to compute gradient of unitary
    ##implemented with trick to make evaluation much faster
    
    ## compute circuit needed for gradient by shifting each parameter one by by one by delta_x
    ## split circuit into   circuit_state*base_circuit
    ##start with circuit_state=Id, and base_circuit= full circuit
    ## adjust the corresponding parameter while going from left to right

    ##circuit to the right
    base_circuit=get_unitary(n_qubits,depth,circuit_params,add_fixed_unitary=add_fixed_unitary,ini_state=ini_state,model=model,opEntangler=opEntangler)

    circuit_orig=qt.Qobj(base_circuit)

    n_circuit_parameters=len(circuit_params)

    circuit_state=ini_state
    counter_params=0
    circuit_grad_list=[]
    for i in range(depth):
        if(model==0):
            rot_op_list1=[]
            ##y rotations
            for j in range(n_qubits):
                param_index=i*n_qubits*2+j
                counter_params+=1
                angle=circuit_params[param_index]
                rot_op=qt.qip.operations.ry(angle)
                rot_op_list1.append(rot_op)
                
                ##get gradient operator
                rot_op_grad=qt.qip.operations.ry(delta_x,n_qubits,j)
                
                ##compute gradient
                circuit_grad=base_circuit*rot_op_grad*circuit_state
                circuit_grad_list.append(circuit_grad)
                            
    
            transfer_op=qt.tensor(rot_op_list1)
            ##compute circuit from left
            circuit_state=transfer_op*circuit_state
            
            ##uncompute circuit to the right
            base_circuit=base_circuit*transfer_op.dag()
            
            ##z rotations
            rot_op_list2=[]
            for j in range(n_qubits):
                param_index=i*n_qubits*2+n_qubits+j
                counter_params+=1
                angle=circuit_params[param_index]
                
                rot_op_grad=qt.qip.operations.rz(delta_x,n_qubits,j)
                circuit_grad=base_circuit*rot_op_grad*circuit_state
                
                circuit_grad_list.append(circuit_grad)
    
                rot_op=qt.qip.operations.rz(angle)
                rot_op_list2.append(rot_op)
    
            transfer_op=qt.tensor(rot_op_list2)
            circuit_state=transfer_op*circuit_state ##add to left
            base_circuit=base_circuit*transfer_op.dag() ##remove from right
                
            transfer_op=opEntangler[i%len(opEntangler)]
            circuit_state=transfer_op*circuit_state
            base_circuit=base_circuit*transfer_op.dag()


        elif(model==1): ##reduced sqrtiswap +z
            rot_op_list1=[]
            for j in range(n_qubits//2):
                param_index=i*(n_qubits//2)+j
                counter_params+=1
                angle=circuit_params[param_index]
                rot_op=qt.qip.operations.rz(angle)
                rot_op_list1.append(rot_op)
                if(2*j+1<n_qubits):##to avoid edge for odd qubit number
                    rot_op_list1.append(qt.qeye(2))
                
                rot_op_grad=qt.qip.operations.rz(delta_x,n_qubits,2*j)
                
                circuit_grad=base_circuit*rot_op_grad*circuit_state
                circuit_grad_list.append(circuit_grad)
                            
    
            transfer_op=qt.tensor(rot_op_list1)
            circuit_state=transfer_op*circuit_state
            base_circuit=base_circuit*transfer_op.dag()
            
            transfer_op=opEntangler[i%len(opEntangler)]
            circuit_state=transfer_op*circuit_state
            base_circuit=base_circuit*transfer_op.dag()


            
    if(counter_params!=n_circuit_parameters):
        raise NameError("Number parameters used does not match")

    return circuit_grad_list,circuit_orig




def get_qfi_eigvals(circuit_parameters,training_states):
    ##computes data quantum Fisher information
    
    delta_x=1.49011612e-08 ##same as used by LBFGS by default
        

    ##get unitary and unitaries shifted by delta_x for each parameter
    grad_unitary_list,circuit_orig=get_unitary_grad(n_qubits,depth,circuit_parameters,add_fixed_unitary=add_fixed_unitary,ini_state=opId,model=model,opEntangler=opEntangler,delta_x=delta_x)


    qfi_training_dm_eigvals=[]
    qfi_training_dm=[]


    n_parameters=len(circuit_parameters)
    
    diff_unitary_list=[]
        
    ##get gradient of unitaries by finite differences
    for q in range(n_parameters):
        diff_unitary=(grad_unitary_list[q]-circuit_orig)/delta_x
        diff_unitary_list.append(diff_unitary)
        
    
    
    ##state that spans space of training states
    rho_training=1/n_training*sum([training_states[k]*training_states[k].dag() for k in range(n_training)])
    rho_eigvals,rho_eigstates=rho_training.eigenstates()
    
    
    ##projector onto space spanned by training states
    projector=0
    for k in range(len(rho_eigvals)):
        if(rho_eigvals[k]>10**-14):
            projector+=rho_eigstates[k]*rho_eigstates[k].dag()
            
    normalised_projector=projector/projector.tr()
    
    
    ##compute qfim
    single_rho_qfi_elements=np.zeros(n_parameters,dtype=np.complex128)
    
    for p in range(n_parameters):
        single_rho_qfi_elements[p]=(circuit_orig.dag()*normalised_projector*diff_unitary_list[p]).tr()
               
    

    qfi_training_dm=np.zeros([n_parameters,n_parameters])
    for q in range(n_parameters):
        for k in range(q,n_parameters):
            qfi_training_dm[q,k]=4*np.real((diff_unitary_list[q].dag()*normalised_projector*diff_unitary_list[k]).tr()-np.conjugate(single_rho_qfi_elements[q])*single_rho_qfi_elements[k])
            
            
    #use fact that qfi matrix is real and hermitian
    for q in range(n_parameters):
        for k in range(q+1,n_parameters):  
            qfi_training_dm[k,q]=qfi_training_dm[q,k]
            
        
    qfi_training_dm_eigvals,qfi_training_dm_eigvecs=np.linalg.eigh(qfi_training_dm)
  

    
    return qfi_training_dm_eigvals,qfi_training_dm
    


def cost_grad(x):
    ##get gradient for optimisation function
    delta_x=1.49011612e-08 ##same as used by LBFGS by default

    grad_list=np.zeros(len(x))
    grad_unitary_list,circuit_orig=get_unitary_grad(n_qubits,depth,x,add_fixed_unitary=add_fixed_unitary,ini_state=opId,model=model,opEntangler=opEntangler,delta_x=delta_x)

    ##compute train error
    cost_train=[]
    for k in range(n_training):
        output_state=circuit_orig*training_states[k]
        fidelity=get_state_fidelity(output_state,training_states_target[k])

        cost_train.append(fidelity)
        
    res=1-np.mean(cost_train)
    
    ##compute gradient of train error
    for q in range(len(x)):
        cost_shift=[]
        encode_unitary=grad_unitary_list[q]
        for k in range(n_training):
            output_state=encode_unitary*training_states[k]
            fidelity=get_state_fidelity(output_state,training_states_target[k])


            cost_shift.append(1-fidelity)
            
        ##use finite difference method to get gradient
        grad_list[q]=(np.mean(cost_shift)-res)/delta_x
        
    return grad_list

      

def callback_opt(x):
    #is called after every iteration of scipy minimize
    ##saves training and test cost
    global callback_training,callback_test,cost_training_list,cost_test_list
    

    cost_training_list.append(callback_training)
    cost_test_list.append(callback_test)


    

def cost(x,in_unitary=[]):
    ##compute cost function for optimization
    global callback_training,callback_test


    if(type(in_unitary)==list):
        encode_unitary=get_unitary(n_qubits,depth,x,add_fixed_unitary=add_fixed_unitary,ini_state=opId,model=model,opEntangler=opEntangler)
    else:
        encode_unitary=in_unitary
        
    ##get train error
    cost_train_states=[]
    for k in range(n_training):
        output_state=encode_unitary*training_states[k]

        fidelity=get_state_fidelity(output_state,training_states_target[k])
            
        #purity_list_all.append(purity_list)
        cost_train_states.append(fidelity)
        
    ##get test error
    cost_test_states=[]
    for k in range(n_test):
        output_test_state=encode_unitary*test_states[k]
        

        fidelity=get_state_fidelity(output_test_state,test_states_target[k])
            
            

        cost_test_states.append(fidelity)
            
        

    ##these variables are given to callback function to be recorded
    callback_training=1-np.mean(cost_train_states)
    callback_test=1-np.mean(cost_test_states)
    
    

    return callback_training



def get_state_fidelity(output_state,target_state):
    ##compute fidelity
    return np.abs(output_state.overlap(target_state))**2



def get_input_state(n_qubits,depth_input,model,type_state):
    ##gets input states for training and test
    ##can choose different types
    

    if(type_state==0): ##haar random
        ini_state=qt.rand_ket_haar(N=2**n_qubits,dims=dims_vec)

    elif(type_state==1):
        ##random product state
        ini_state=qt.tensor([qt.rand_ket_haar(N=2) for i in range(n_qubits)])
        
        
    elif(type_state==2): ##random states with particle number conservation

        ini_state=qt.tensor([qt.basis(2,1) for i in range(n_particles)]+[qt.basis(2,0) for i in range(n_qubits-n_particles)])

        if(depth_input>0): ##randomize particle number states by applying random particle number convserving unitary U_XY
            n_input_params=get_circuit_parameters(depth_input,n_qubits,1)
            input_circuit_params=rng.random(n_input_params)*2*np.pi
            ini_state=get_unitary(n_qubits,depth_input,input_circuit_params,add_fixed_unitary=[],ini_state=ini_state,model=1,opEntangler=opEntangler)
        


    return ini_state


cost_train_final_list = []
cost_test_final_list = []
rank_DQFIM_list = []

noise_type = "amplitude"  # Choose the type of noise: 'amplitude', 'phase', or 'bit_flip'
noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
cost_train_final_lists = {noise_ratio: [] for noise_ratio in noise_levels}
cost_test_final_lists = {noise_ratio: [] for noise_ratio in noise_levels}
rank_DQFIM_lists = {noise_ratio: [] for noise_ratio in noise_levels}


for noise_ratio in noise_levels:
    print(f"=== Processing with noise ratio: {int(noise_ratio * 100)}% and of type: {str(noise_type)} ===")

    cost_train_final_list = []
    cost_test_final_list = []
    rank_DQFIM_list = []

    for p in range(datapoints):
        n_training = n_training_list[p]
        print("Number training states", n_training)

        levels = 2
        opZ = [genFockOp(qt.sigmaz(), i, n_qubits, levels) for i in range(n_qubits)]
        opX = [genFockOp(qt.sigmax(), i, n_qubits, levels) for i in range(n_qubits)]
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

        training_states = []
        training_states_target = []

        for k in range(n_training):
            ini_state = get_input_state(n_qubits, depth_input, model, type_state)
            training_states_target.append(ini_state)
            state_data = data_unitary * ini_state
            training_states.append(state_data)

        training_states_target = inject_label_noise(training_states_target, noise_ratio, noise_type=noise_type, n_qubits=n_qubits)
        test_states = []
        test_states_target = []

        for k in range(n_test):
            ini_state = get_input_state(n_qubits, depth_input, model, type_state_test)
            test_states_target.append(ini_state)
            state_data = data_unitary * ini_state
            test_states.append(state_data)

        test_states_target = inject_label_noise(test_states_target, noise_ratio, noise_type=noise_type, n_qubits=n_qubits)

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
        qfi_training_dm_eigvals, qfi_training_dm = get_qfi_eigvals(xr, training_states)

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
plt.savefig("figure/OC/noise/Rank_DQFIM_vs_training_states_noise_levels.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot train cost vs L for different noise levels
plt.figure(figsize=(10, 6))
for noise_ratio, cost_train_final_list in cost_train_final_lists.items():
    plt.plot(n_training_list, cost_train_final_list, label=f"Noise: {int(noise_ratio * 100)}%")
plt.ylabel('Train cost')
plt.xlabel("L")
plt.title("Train cost vs number of training states for different noise levels")
plt.legend(loc="upper right")
plt.savefig("figure/OC/noise/Train_cost_vs_training_states_noise_levels.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot test cost vs L for different noise levels
plt.figure(figsize=(10, 6))
for noise_ratio, cost_test_final_list in cost_test_final_lists.items():
    plt.plot(n_training_list, cost_test_final_list, label=f"Noise: {int(noise_ratio * 100)}%")
plt.ylabel('Test cost')
plt.xlabel("L")
plt.title("Test cost vs number of training states for different noise levels")
plt.legend(loc="upper right")
plt.savefig("figure/OC/noise/Test_cost_vs_training_states_noise_levels.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot generalization vs L for different noise levels
plt.figure(figsize=(10, 6))
for noise_ratio, cost_test_final_list in cost_test_final_lists.items():
    plt.plot(n_training_list, np.array(cost_train_final_list) - np.array(cost_test_final_list), label=f"Noise: {int(noise_ratio * 100)}%")
plt.ylabel('Generalization')
plt.xlabel("L")
plt.title("Generalization vs number of training states for different noise levels")
plt.legend(loc="upper right")
plt.savefig("figure/OC/noise/Generalization_vs_training_states_noise_levels.png", dpi=300, bbox_inches='tight')
plt.show()





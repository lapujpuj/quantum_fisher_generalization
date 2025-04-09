
import numpy as np

import qutip as qt
import scipy
import scipy.io
import operator

import matplotlib.pyplot as plt
from functools import reduce




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




def get_qfi_eigvals(circuit_parameters,training_states,test_states,n_qubits,depth,add_fixed_unitary,opId,opEntangler,model=0):
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

    n_training=len(training_states)
    n_test=len(test_states)
        
    
    
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
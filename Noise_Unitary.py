from qutip import Qobj, basis, qeye
from qutip import  tensor


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
    


from qutip import  tensor






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
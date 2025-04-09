import numpy as np
import h5py
from tenpy.models.spins import SpinChain
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
import qutip as qt
from qutip import Qobj

rng = np.random.default_rng()




def mps_to_qutip(mps, is_ket=True):
    """
    Convertit un tenpy.networks.mps.MPS en objet Qutip Qobj
    
    Args:
        mps: L'état MPS de TeNPy (tenpy.networks.mps.MPS)
        is_ket: True si l'état doit être un ket, False pour une matrice densité
        
    Returns:
        Un objet Qutip Qobj représentant l'état quantique
    """
    # Obtenir le tenseur complet
    theta = mps.get_theta(0, mps.L).to_ndarray()
    
    # Éliminer les dimensions des indices de liaison (premier et dernier)
    # et conserver uniquement les dimensions physiques
    if theta.shape[0] == 1 and theta.shape[-1] == 1:
        state_tensor = theta[0, ..., 0]  # Éliminer les dimensions des indices de liaison
    
    # Aplatir en vecteur
    full_state = state_tensor.reshape(-1)
    
    # Déterminer les dimensions locales des sites
    local_dims = [site.dim for site in mps.sites]
    
    if is_ket:
        # Créer un ket Qutip
        return qt.Qobj(full_state, dims=[local_dims, [1]*len(local_dims)])
    else:
        # Créer une matrice densité Qutip
        ket = qt.Qobj(full_state, dims=[local_dims, [1]*len(local_dims)])
        return ket * ket.dag()



def get_ground_state(L, J, h, r = .5):
    model_params = {
        'S': 0.5,
        'L': L,
        'Jx': -J * (1 + r),
        'Jy': -J * (1 - r),
        'Jz': 0.0,
        'hz': h,
        'bc_MPS': 'finite',
        'conserve': None
        }


    model = SpinChain(model_params)
    psi0 = MPS.from_product_state(model.lat.mps_sites(), ["up"] * L, "finite")


    dmrg_params = {
        'mixer': True,
        'max_sweeps': 10,
        'trunc_params': {'chi_max': 100}
    }

    eng = dmrg.TwoSiteDMRGEngine(psi0, model, dmrg_params)
    E, psi = eng.run()

    # Convertir l'état MPS en Qobj
    psi = mps_to_qutip(psi)

    return psi, E

# Calcul des observables (ici aimantation selon Z)
def magnetization_z(psi):
    """Calcul de l'aimantation selon Z"""
    L = len(psi.dims[0])
    mz = 0
    for i in range(L):
        mz += qt.expect(qt.sigmaz(), psi.ptrace(i))
    return mz / L



def generate_samples(n_qubit, n_sample, r=0.5, gammas=[[0, 0.5], [1.5, 2]], class_balance=0.5, h =1):
    num_samples_class_0 = int(n_sample * class_balance)
    num_samples_class_1 = n_sample - num_samples_class_0

    gammas_0 = np.random.uniform(gammas[0][0], gammas[0][1], num_samples_class_0)
    gammas_1 = np.random.uniform(gammas[1][0], gammas[1][1], num_samples_class_1)

    X_features = []
    X = []  # Liste pour stocker les états quantiques
    y_labels = []

    # Classe 0 (paramagnétique)
    for gamma in gammas_0:
        J  =  gamma * h  # h fixé à 1, J = gamma * h
        psi, E = get_ground_state(n_qubit, J, h, r)
        mz = magnetization_z(psi)
        X.append(psi)  # Ajouter l'état quantique à la liste
        X_features.append([gamma, mz])
        y_labels.append(0)

    # Classe 1 (ferromagnétique)
    for gamma in gammas_1:
        J= gamma*h
        psi, E = get_ground_state(n_qubit, J, h, r)
        mz = magnetization_z(psi)
        X.append(psi)
        X_features.append([gamma, mz])
        y_labels.append(1)

    X_features = np.array(X_features)
    y_labels = np.array(y_labels)

    return (X, X_features), y_labels

def get_sample_from_dataset(X, y, n_sample):
    """
    Récupère un échantillon à partir du jeu de données.
    """
    indices = np.random.choice(len(X), n_sample, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    return X_sample, y_sample

def create_dataset_in_file(n_qubit, n_sample, filename ="data/phase_classification_data",  r=0.5, gammas=[[0, 0.5], [1.5, 2]], class_balance=0.5):
    """
    Crée un jeu de données et l'enregistre dans un fichier HDF5.
    """
    X_features, y_labels = generate_samples(n_qubit, n_sample, r, gammas, class_balance)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('X', data=X_features)
        f.create_dataset('y', data=y_labels)
    print(f"Dataset saved to {filename}")


def get_sample_from_file(n_sample = 100, filename = "data/phase_classification_data"):
    """
    Récupère un échantillon à partir d'un fichier HDF5.
    """
    with h5py.File(filename, 'r') as f:
        X = f['X'][:]
        y = f['y'][:]

    return get_sample_from_dataset(X, y, n_sample)

def inject_label_noise(states_target, noise_ratio):
    """Inject noise into a fraction of the training labels."""
    noisy_target = states_target.copy()
    n_total = len(noisy_target)
    n_noisy = int(noise_ratio * n_total)
    idx_to_corrupt = rng.choice(n_total, n_noisy, replace=False)
    for i in idx_to_corrupt:
        # Replace the label with a randomly chosen one (different from current)
        noisy_target[i] = 1 - noisy_target[i]
    return noisy_target
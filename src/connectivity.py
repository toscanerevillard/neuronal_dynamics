import numpy as np
from scipy.sparse import csr_matrix

# Purpose:
# Generate the recurrent connectivity matrix W for the random sparse E/I network
# required for Exercise 1.1

# Source of Code and Justification:
# The project defines that each neuron receives exactly
# KE = p*NE excitatory and KI = p*NI inhibitory incoming connections,
# sampled randomly (sparse).
# Synaptic strengths follow the project’s E/I weight scheme:
#   E synapses: +J   (to both E and I postsynaptic neurons)
#   I synapses: -gJ  (to both E and I postsynaptic neurons)
# The project also recommends scipy.sparse.csr_matrix because each neuron has
# only few incoming connections.

# Approach:
# - Fix an index convention : first NE neurons are E (excitatory), 
#   remaining NI are I (inhibitory).
# - For each postsynaptic neuron i, independently sample KE distinct presynaptic E
#   indices and KI distinct presynaptic I indices.
# - Set matrix entries W[i,j] to J or -gJ depending on presynaptic type.
# - Convert the resulting matrix to CSR format with scipy.sparse.csr_matrix.

# What to say in Report:
# “We constructed a sparse random recurrent connectivity matrix with fixed
# KE excitatory and KI inhibitory inputs per neuron, as specified in
# Exercise 1.1. Excitatory synapses were assigned strength +J and inhibitory
# synapses -gJ. The connectivity was stored as a CSR sparse matrix for efficient
# recurrent current computation in large networks.”


def generate_sparse_connectivity(NE, NI, KE, KI, J, g, seed=None):
    """
    Build a random sparse weight matrix W of shape (N, N) with N = NE + NI.

    Neuron indices 0 to NE-1 are excitatory; NE to N-1 are inhibitory.
    Each postsynaptic neuron receives exactly KE excitatory and KI inhibitory
    incoming connections, sampled uniformly at random.

    Weights according to the project convention:
        w{E<-E} = w{I<-E} = J
        w{E<-I} = w{I<-I} = -g * J

    Parameters (inputs)
    ----------
    NE, NI : int
        Population sizes
    KE, KI : int
        Fixed numbers of excitatory / inhibitory inputs per neuron
    J : float
        Excitatory synaptic strength (pC)
    g : float
        Inhibitory / excitatory strength ratio
    seed : int (optional), RNG seed for reproducible connectivity

    Returns (outputs)
    -------
    W : scipy.sparse.csr_matrix -> Sparse connectivity matrix
    """
    rng = np.random.default_rng(seed)
    N = NE + NI

    index_E = np.arange(NE)
    index_I = np.arange(NE, N)

    W = np.zeros((N, N), dtype=float)

    for i in range(N):
        presynaptic_exc = rng.choice(index_E, size=KE, replace=False)
        presynaptic_inh = rng.choice(index_I, size=KI, replace=False)
        W[i, presynaptic_exc] = J
        W[i, presynaptic_inh] = -g * J

    return csr_matrix(W)

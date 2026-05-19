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



# Purpose:
# Generate the population-level unit-unit connectivity matrix W for the cortical field model.
# This matrix describes how the E and I populations of different E-I units interact across space.

# Source of Code and Justification:
# Exercise 2.3 asks us to implement generate_unit_connectivity(Nunits, sigma, W0, g, gamma).
# The project defines a ring of Nunits E-I units with positions x_alpha equally spaced in [0, 1].
# Interactions depend on ring distance:
#   W^{E<-E}_{alpha,beta} = W0 * f(x_alpha, x_beta)
#   W^{I<-E}_{alpha,beta} = g * gamma * W0 * (1 - f(x_alpha, x_beta))
#   W^{E<-I}_{alpha,beta} = W^{I<-I}_{alpha,beta} = 0
# where f = 1 if the ring distance is <= sigma, and 0 otherwise.
# Self-connections between units are excluded because within-unit interactions are already handled
# by each EIUnit's recurrent sparse network.

# Approach:
# - Place Nunits units uniformly on a ring between 0 and 1.
# - For every pair of units alpha and beta, compute the periodic ring distance.
# - If alpha == beta, skip the connection.
# - If beta is close to alpha, add short-range excitation from E_beta to E_alpha.
# - If beta is far from alpha, add long-range excitation from E_beta to I_alpha.
# - Store the result in a 2*Nunits by 2*Nunits matrix.
# - Use the index convention:
#       0 ... Nunits-1           = excitatory populations
#       Nunits ... 2*Nunits-1    = inhibitory populations

# What to say in Report:
# "We constructed the unit-unit connectivity matrix on a one-dimensional ring with periodic
# boundary conditions. Nearby excitatory populations projected to excitatory populations,
# implementing short-range excitation. Distant excitatory populations projected to inhibitory
# populations, implementing effective long-range inhibition under the constraint that long-range
# projections originate from excitatory neurons. Inhibitory populations did not project between
# units. Self-connections were removed because local recurrent E/I interactions were already
# included inside each EIUnit."

import numpy as np


def ring_distance(x_alpha, x_beta):
    """
    Compute the shortest distance between two positions on a ring of circumference 1.
    """
    raw_distance = abs(x_alpha - x_beta)
    return min(raw_distance, 1.0 - raw_distance)


def generate_unit_connectivity(Nunits, sigma, W0, g, gamma):
    """
    Generate the unit-unit connectivity matrix for the cortical field model.

    Parameters
    ----------
    Nunits : int
        Number of E-I units on the ring.
    sigma : float
        Spatial range of short-range excitatory interactions.
    W0 : float
        Strength of unit-unit coupling.
    g : float
        Inhibitory/excitatory strength ratio from the recurrent E-I network.
    gamma : float
        Ratio NI / NE.

    Returns
    -------
    W_unit : ndarray, shape (2*Nunits, 2*Nunits)
        Population-level connectivity matrix.
        Rows are target populations, columns are source populations.
    x : ndarray, shape (Nunits,)
        Positions of the E-I units on the ring.
    """

    x = np.linspace(0.0, 1.0, Nunits, endpoint=False)

    W_unit = np.zeros((2 * Nunits, 2 * Nunits), dtype=float)

    for alpha in range(Nunits):          # target unit
        for beta in range(Nunits):       # source unit

            # Exclude unit self-connections.
            if alpha == beta:
                continue

            d = ring_distance(x[alpha], x[beta])

            target_E = alpha
            target_I = Nunits + alpha
            source_E = beta

            if d <= sigma:
                # Short-range excitation: E_beta -> E_alpha
                W_unit[target_E, source_E] = W0
            else:
                # Long-range effective inhibition: E_beta -> I_alpha
                W_unit[target_I, source_E] = g * gamma * W0

    return W_unit, x
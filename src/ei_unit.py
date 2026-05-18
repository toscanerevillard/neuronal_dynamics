import numpy as np
from src.lif import lif_euler_step

# Purpose :
# Simulate a recurrent sparse E/I network using the same discrete-time LIF model as Exercise 0,
# with recurrent synaptic input implemented from delayed presynaptic spike trains

# Source of Code and Justification :
# Exercise 1 defines the postsynaptic current as
# I(t) = sum_j{w_ij*s_j(t - tau_delay)} + I_ext(t) + I_bg(t),
# according to the project's sparse connectivity matrix W and synaptic transmission delay tau_delay.
# The stochastic background input follows the Exercise 0 Poisson formulation 
# (included for consistency with all subsequent exercises in the assignment).
# The discrete spike convention follow again the Exercise 0: S[k,i] = 1/dt 
# VERY IMPORTANT : The argument I_ext_t is a matrix of shape (n_steps, N) used for ex 1.5
# used in Exercise 1.5

# Approach:
# - Describe total number of neurons N from W.shape and split inhibitory neurons as N - NE 
#   (index convention : first NE neurons are excitatory, remaining are inhibitory).
# - Maintain membrane potentials across time steps with lif_euler_step (forward Euler integration).
# - At each timestep, compute delayed spikes s_delayed from S[k - delay_steps].
# - Compute recurrent synaptic current as I_syn = W @ s_delayed, directly implementing Eq. (3):
#   w_ij (pC) * s_j (1/ms) = I_syn (nA).
# - Combine inputs as I_syn + uniform constant external current + independent Poisson background.
# - Threshold resets voltages and records spikes; store trajectories U and spike matrix S.

# What to say in Report:
# "We simulated the recurrent E/I network using the full input decomposition required in Exercise 1:
# delayed recurrent currents computed as W @ s_delayed, directly implementing Eq. (3) where
# w_ij (pC) multiplied by s_j = 1/dt (ms^-1) gives a synaptic current in nA, plus external
# current and stochastic background fluctuations.
# Spike generation used the exercise's discrete spike-train
# convention, and neuron dynamics reused the Euler-integrated LIF update from Exercise 0."

def simulate_population_exc_inh(
    W,
    NE,
    I0_const_nA,
    T_ms,
    dt,
    tau_m,
    R,
    theta,
    u_reset,
    tau_delay_ms,
    n_bg,
    bg_scale,
    *,
    rng,
    u0=None,
    I_ext_t=None,
):
    """
    Total input:
      I_total = W @ S[k-delay] + I0_const + I_Poisson_background
    Returns U, S for N = NE + NI from W.shape[0].
    Spike convention respect part 0 :
      S[k, i] = 1/dt on the step neuron i spikes, else S[k, i] = 0
    """

    NI = int(W.shape[0] - NE)
    N = int(W.shape[0])

    n_steps = int(round(T_ms / dt))
    delay_steps = int(round(tau_delay_ms / dt))

    if u0 is None:
        u = rng.uniform(u_reset, theta, size=N).astype(float)
    else:
        u = np.asarray(u0, dtype=float).copy()

    U = np.zeros((n_steps, N), dtype=float)
    S = np.zeros((n_steps, N), dtype=float)

    for k in range(n_steps):

        if k >= delay_steps:
            s_delayed = S[k - delay_steps]
        else:
            s_delayed = np.zeros(N, dtype=float)

        # Compute recurrent synaptic current directly from Eq. (3):
        # w_ij (pC) * s_j (1/ms) = I_syn (nA).
        I_syn = W @ s_delayed
        
        if n_bg > 0:
            I_bg = bg_scale * rng.poisson(n_bg, size=N)
        else:
            I_bg = 0.0

        if I_ext_t is not None:
            I_ext = I_ext_t[k]
        else:
            I_ext = np.full(N, float(I0_const_nA), dtype=float)

        I_total = I_syn + I_ext + I_bg

        u = lif_euler_step(u, I_total, dt=dt, tau_m=tau_m, R=R)

        spiked = u >= theta
        S[k, spiked] = 1.0 / dt
        u[spiked] = u_reset

        U[k] = u

    return U, S


# Purpose:
# Compute a scalar mean firing rate (Hz) for a selected neuron subset (population or slice)
# by averaging spikes over a prescribed time intervall.

# Source of Code and Justification:
# Exercise 1.2 follows Exercise 0.2: firing rates should be summarized from spikes measured over a
# fixed time interval (final 50 ms of a 100 ms run).
# With the Exercise 0 discrete spike convention S[k,i] = 1/dt on spike steps, multiplying 
# by dt counts spikes: sum_k{S[k,i]*dt} equals the total number of spikes of neuron i in the window.

# Approach:
# - Convert time bounds from milliseconds into discrete timestep indices with round(t_ms / dt).
# - Restrict S to neurons in index_interval along axis 1.
# - Integrate spikes per neuron across the temporal interval.
# - Convert window length from timesteps -> seconds as (t1-t0)*dt/1000.
# - Return population mean spikes per neuron per second by averaging counts 
#   and dividing by duration_s.

# What to say in Report:
# "Population firing rates were estimated by averaging spiking activity across neurons in each 
# population, restricting to the analysis interval indicated in the Exercise 0.2. Spike counts 
# followed the assignment discrete-time convention by integrating S*(dt), and firing rates were 
# reported in Hz by dividing by interval duration."


def population_rate_hz(S, index_interval, dt, t_start_ms, t_end_ms):
    """
    Mean firing rate in Hz for neurons in the desired interval along axis 1,
    averaged over [t_start_ms, t_end_ms].
    Requires S=1/dt on spike, dt in ms.
    Spike count for one neuron integrates as sum_k S[k,i]*dt equals to the number of spikes.
    """
    t0 = int(round(t_start_ms / dt))
    t1 = int(round(t_end_ms / dt))

    interval = S[t0:t1, index_interval]  
    counts = interval.sum(axis=0) * float(dt)
    duration_s = (t1 - t0) * float(dt) / 1000.0
    mean_hz = counts.mean() / duration_s

    return float(mean_hz)


# Purpose:
# Define a reusable EIUnit class representing one recurrent excitatory-inhibitory (E-I) unit.
# This class wraps the Exercise 1 sparse recurrent E/I network into an object that can be
# updated one timestep at a time using a step(I_E, I_I) method.
# The class will serve as the fundamental building block for the cortical field model in Exercise 2.

# Source of Code and Justification:
# Exercise 2.2 explicitly suggests wrapping the Exercise 1 network into a class defining
# a single E-I unit with a function step(I_E, I_I).
# The recurrent within-unit dynamics are reused directly from Exercise 1:
# - sparse recurrent connectivity matrix W
# - delayed synaptic interactions
# - stochastic background input
# - Euler integration of LIF neurons
#
# The Neuronal Dynamics field-model framework also represents cortex as interacting local
# population units distributed across space. Each EIUnit therefore represents one local
# cortical population pair that will later be coupled to other units through population-level
# interactions on the ring.

# Approach:
# - Store all neuron and simulation parameters inside the class.
# - Initialize membrane potentials and delayed spike-history buffers.
# - Reuse the Exercise 1 recurrent synaptic dynamics inside the unit.
# - Implement a step(I_E, I_I) method that:
#     1. computes delayed recurrent synaptic input,
#     2. adds external and stochastic background inputs,
#     3. updates membrane voltages using Euler integration,
#     4. detects threshold crossings and resets spiking neurons,
#     5. stores the spike vector for delayed recurrence,
#     6. returns instantaneous E and I population activities.
#
# This step-based architecture allows many E-I units to be simulated independently and later
# coupled together in the cortical field model.

# What to say in Report:
# "To construct the cortical field model, the recurrent E/I network from Exercise 1 was wrapped
# into an EIUnit class representing one local cortical population pair. Each EIUnit contains
# sparse recurrent excitatory and inhibitory connectivity, delayed synaptic interactions,
# stochastic background activity, and LIF membrane dynamics. The class exposes a
# step(I_E, I_I) method that advances the network by one timestep and returns the
# instantaneous excitatory and inhibitory population activities. This modular structure
# allows multiple E-I units to be coupled together spatially in the ring model."


class EIUnit:
    """
    One recurrent E-I unit used as the building block of the cortical field model.

    The unit contains:
    - NE excitatory neurons
    - NI inhibitory neurons
    - one sparse recurrent connectivity matrix W
    - one-step LIF dynamics with delayed recurrent synaptic input

    The step(I_E, I_I) method updates the unit by one timestep.
    """

    def __init__(
        self,
        W,
        NE,
        dt,
        tau_m,
        R,
        theta,
        u_reset,
        tau_delay_ms,
        n_bg,
        bg_scale,
        rng,
        u0=None,
    ):
        self.W = W
        self.NE = int(NE)
        self.N = int(W.shape[0])
        self.NI = self.N - self.NE

        self.dt = float(dt)
        self.tau_m = float(tau_m)
        self.R = float(R)
        self.theta = float(theta)
        self.u_reset = float(u_reset)
        self.tau_delay_ms = float(tau_delay_ms)
        self.delay_steps = int(round(tau_delay_ms / dt))

        self.n_bg = float(n_bg)
        self.bg_scale = float(bg_scale)
        self.rng = rng

        if u0 is None:
            self.u = self.rng.uniform(self.u_reset, self.theta, size=self.N).astype(float)
        else:
            self.u = np.asarray(u0, dtype=float).copy()

        # Keep spike history so delayed recurrent input can be computed.
        self.spike_history = []

    def step(self, I_E, I_I):
        """
        Advance this E-I unit by one timestep.

        Parameters
        ----------
        I_E : float
            External input to all excitatory neurons of this unit.
        I_I : float
            External input to all inhibitory neurons of this unit.

        Returns
        -------
        u : array, shape (N,)
            Updated membrane potentials.
        s : array, shape (N,)
            Spike vector for this timestep, with s[i] = 1/dt if neuron i spikes.
        r_E : float
            Instantaneous excitatory population activity.
        r_I : float
            Instantaneous inhibitory population activity.
        """

        if len(self.spike_history) >= self.delay_steps:
            s_delayed = self.spike_history[-self.delay_steps]
        else:
            s_delayed = np.zeros(self.N, dtype=float)

        # Compute recurrent synaptic current directly from Eq. (3):
        # w_ij (pC) * s_j (1/ms) = I_syn (nA).
        I_syn = self.W @ s_delayed

        if self.n_bg > 0:
            I_bg = self.bg_scale * self.rng.poisson(self.n_bg, size=self.N)
        else:
            I_bg = 0.0

        I_ext = np.zeros(self.N, dtype=float)
        I_ext[:self.NE] = float(I_E)
        I_ext[self.NE:] = float(I_I)

        I_total = I_syn + I_ext + I_bg

        self.u = lif_euler_step(
            self.u,
            I_total,
            dt=self.dt,
            tau_m=self.tau_m,
            R=self.R,
        )

        spiked = self.u >= self.theta

        s = np.zeros(self.N, dtype=float)
        s[spiked] = 1.0 / self.dt

        self.u[spiked] = self.u_reset

        self.spike_history.append(s.copy())

        r_E = s[:self.NE].mean()
        r_I = s[self.NE:].mean()

        return self.u.copy(), s, r_E, r_I
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

# Approach:
# - Describe total number of neurons N from W.shape and split inhibitory neurons as N - NE 
#   (index convention : first NE neurons are excitatory, remaining are inhibitory).
# - Maintain membrane potentials across time steps with lif_euler_step (forward Euler integration).
# - At each timestep, compute delayed spikes s_delayed from S[k - delay_steps], 
#   then Isyn = W @ s_delayed.
# - Combine inputs as I_syn + uniform constant external current + independent Poisson background.
# - Threshold resets voltages and records spikes; store trajectories U and spike matrix S.

# What to say in Report:
# “We simulated the recurrent E/I network using the full input decomposition required in Exercise 1:
# delayed recurrent currents from sparse synaptic weights plus constant external current and 
# stochastic background fluctuations. Spike generation used the exercise’s discrete spike-train 
# convention, and neuron dynamics reused the Euler-integrated LIF update from Exercise 0.”

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
):
    """
    Total input:
      I_total = W * S[k-delay] + I0_const + I_Poisson_background
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

        I_syn = W @ s_delayed

        if n_bg > 0:
            I_bg = bg_scale * rng.poisson(n_bg, size=N)
        else:
            I_bg = 0.0

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
# “Population firing rates were estimated by averaging spiking activity across neurons in each 
# population, restricting to the analysis interval indicated in the Exercise 0.2. Spike counts 
# followed the assignment discrete-time convention by integrating S*(dt), and firing rates were 
# reported in Hz by dividing by interval duration.”


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
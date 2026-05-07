import numpy as np


# Purpose:
# Implement one time-step update of the LIF neuron using forward Euler integration.

# Source of Code and Justification:
# The project gives the LIF differential equation:
# τ_m du/dt = -u + R I(t)
# and explicitly states it should be integrated using the forward Euler method:
# u(t+Δt) = u(t) + Δt * du/dt
# The key computational pattern is:
# simulate LIF dynamics via Euler updates.
# This corresponds exactly to the discretized equation used in the notebook of week 1.

# Approach:
# - Compute the change in voltage using the Euler rule:
#   u_next = u + (dt/tau_m) * (-u + R * I)
# - This function only updates subthreshold voltage
# - Thresholding and reset are handled outside this function

# What to say in Report:
# “The membrane potential is updated using forward Euler integration of the
# passive membrane equation. This discretizes the continuous LIF dynamics
# into time steps suitable for numerical simulation.”

def lif_euler_step(u, I, dt=0.5, tau_m=20.0, R=1.0):
    return u + (dt / tau_m) * (-u + R * I)

# Purpose:
# Import the reusable LIF simulation function from src/lif.py.

# Source of Code and Justification:
# The project asks us to write a method that evolves the membrane potentials
# and returns both membrane potentials and spikes through time.
# It also defines the discrete spike train convention:
# s_i(t_k) = 1/Δt when the neuron spikes, and zero otherwise.
# The notebook from week 1 has the computational
# pattern: define parameters, simulate LIF dynamics, record spikes, and compute rates.

# Approach:
# - Keep the simulator in src/lif.py so it can be reused in later exercises.
# - At each time step, compute total input current.
# - Optionally add independent Poisson background input to each neuron.
# - Update membrane potentials using the Euler LIF step.
# - Detect threshold crossings.
# - Store spikes as 1/dt and reset spiking neurons to u_reset.

# What to say in Report:
# “We implemented a reusable population simulator for unconnected LIF neurons.
# At each time step, membrane potentials are updated with Euler integration,
# spikes are recorded using the assignment's discrete spike-train convention,
# and neurons crossing threshold are reset.”

def simulate_lif_population(
    I_ext,
    u0,
    dt=0.5,
    tau_m=20.0,
    R=1.0,
    theta=20.0,
    u_reset=-10.0,
    n_bg=0.0,
    bg_scale=1.0,
    seed=None,
):
    """
    Simulate an unconnected population of LIF neurons.

    Parameters
    ----------
    I_ext : array, shape (n_steps,) or (n_steps, N)
        External input current in nA.
    u0 : array, shape (N,)
        Initial membrane potentials in mV.

    Returns
    -------
    U : array, shape (n_steps, N)
        Membrane potentials in mV.
    S : array, shape (n_steps, N)
        Spike trains. S[k, i] = 1/dt if neuron i spikes at time step k.
    """
    rng = np.random.default_rng(seed)

    u = np.asarray(u0, dtype=float).copy()
    I_ext = np.asarray(I_ext, dtype=float)

    N = u.size

    if I_ext.ndim == 1:
        I_ext = I_ext[:, None] * np.ones((1, N))

    n_steps = I_ext.shape[0]

    U = np.zeros((n_steps, N))
    S = np.zeros((n_steps, N))

    for k in range(n_steps):
        I_total = I_ext[k].copy()

        if n_bg > 0:
            I_bg = bg_scale * rng.poisson(n_bg, size=N)
            I_total += I_bg

        u = lif_euler_step(u, I_total, dt=dt, tau_m=tau_m, R=R)

        spiked = u >= theta
        S[k, spiked] = 1.0 / dt
        u[spiked] = u_reset

        U[k] = u

    return U, S

# Purpose:
# Import the theoretical deterministic f-I curve for LIF neurons.

# Source of Code and Justification:
# Exercise 0.2 asks us to compare the simulated f-I curve to the theoretical
# f-I curve for LIF neurons without background input.
# The LIF lecture derives the gain function/f-I curve for constant input:
# firing rate as a function of input strength.
# From the notebook 1:  minimal-current and f-I curve
# sections as directly relevant to Exercise 0.2.

# Approach:
# - Use the reset-aware theoretical LIF firing-rate formula.
# - Set firing rate to zero when the constant input is below threshold,
#   because the neuron cannot reach threshold deterministically.

# What to say in Report:
# “The analytic f-I curve describes the deterministic LIF response to constant
# input without stochastic background. It predicts zero firing below threshold
# and increasing firing rate for stronger suprathreshold currents.”

def lif_theoretical_fi(I_values, tau_m=20.0, R=1.0, theta=20.0, u_reset=-10.0):
    """
    Theoretical f-I curve for a deterministic LIF neuron without background input.

    Parameters
    ----------
    I_values : array
        Constant input currents in nA.

    Returns
    -------
    f : array
        Firing rates in Hz.
    """
    I_values = np.asarray(I_values, dtype=float)

    f = np.zeros_like(I_values)
    mask = R * I_values > theta

    eps = 1e-9

    num = (R * I_values[mask] - u_reset)
    denom = (R * I_values[mask] - theta)

    denom = np.maximum(denom, eps)  # avoid division by zero or log explosion

    T_ms = tau_m * np.log(num / denom)

    f[mask] = 1000.0 / T_ms

    return f
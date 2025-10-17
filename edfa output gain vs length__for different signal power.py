import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical constants ---
h = 6.626e-34    # Planck constant (J*s)
c = 3e8         # speed of light (m/s)

# --- EDFA parameters 
N_total = 1e25      # Erbium ion density (m^-3)
sigma_sa = 2.14e-25    # signal absorption cross-section at 1550 nm (m^2)
sigma_se = 3.8e-25    # signal emission cross-section at 1550 nm (m^2)
sigma_pa = 1.8e-25    # pump absorption cross-section at 980 nm (m^2)
sigma_pe = 3.14e-25        # pump emission cross-section (often small, set 0 for 980 nm pumping)
tau = 10e-3           # excited state lifetime (s)
A_eff = 50e-12        # effective mode area (m^2) ~ 50 micrometer^2 typical
Gamma_s = 0.8         # overlap factor for signal
Gamma_p = 0.9         # overlap factor for pump
alpha_s = 0     # background loss for signal (m^-1) -> 0.2 dB/km ~ converted (here small)
alpha_p = 0     # background loss for pump (m^-1)

# Wavelengths / photon energies
wl_s = 1550e-9
wl_p = 980e-9
nu_s = c / wl_s
nu_p = c / wl_p
hnu_s = h * nu_s
hnu_p = h * nu_p

# Input signal power (set small so gain is visible)
P_sig_in = 1e-3  # 1 mW input signal

# Fiber length to simulate
L = 10.0       # meters
z_eval = np.linspace(0, L, 1001)

# Function to compute steady-state excited population fraction N2 (approximate)
def compute_N2(Pp, Ps):
    """
    Compute the excited-state population N2 (m^-3) using steady-state balance of excitation and decay.
    This uses approximate stimulated transition rates:
       W_p_abs = Gamma_p * sigma_pa * Pp / (h*nu_p*A_eff)
       W_s_abs = Gamma_s * sigma_sa * Ps / (h*nu_s*A_eff)
       W_p_em  = Gamma_p * sigma_pe * Pp / (h*nu_p*A_eff)
       W_s_em  = Gamma_s * sigma_se * Ps / (h*nu_s*A_eff)
    Then N2 = N_total * (W_p_abs + W_s_abs) / (W_p_abs + W_p_em + W_s_abs + W_s_em + 1/tau)
    (This is a common simple steady-state approximation; it neglects ASE and forward/backward signal splitting.)
    """
    Wp_abs = 0.0 if Pp <= 0 else (Gamma_p * sigma_pa * Pp) / (hnu_p * A_eff)
    Wp_em  = 0.0 if Pp <= 0 else (Gamma_p * sigma_pe * Pp) / (hnu_p * A_eff)
    Ws_abs = 0.0 if Ps <= 0 else (Gamma_s * sigma_sa * Ps) / (hnu_s * A_eff)
    Ws_em  = 0.0 if Ps <= 0 else (Gamma_s * sigma_se * Ps) / (hnu_s * A_eff)
    denom = Wp_abs + Wp_em + Ws_abs + Ws_em + 1.0 / tau
    if denom <= 0:
        return 0.0
    N2 = N_total * (Wp_abs + Ws_abs) / denom
    return float(np.clip(N2, 0.0, N_total))

# ODE system: coupled pump and signal power equations dP/dz
def edfa_odes(z, y):
    Pp, Ps = y
    # Compute N2 from current local powers
    N2 = compute_N2(Pp, Ps)
    N1 = N_total - N2
    # Signal gain coefficient (m^-1)
    g_s = Gamma_s * (sigma_se * N2 - sigma_sa * N1)
    # Pump absorption (we neglect pump stimulated emission for 980 nm here)
    att_p = Gamma_p * sigma_pa * N1
    # Differential equations
    dPp_dz = - (att_p + alpha_p) * Pp
    dPs_dz = g_s * Ps - alpha_s * Ps
    return [dPp_dz, dPs_dz]

# Function to run simulation for a given input pump power (in W)
def run_sim(Pp_in):
    y0 = [Pp_in, P_sig_in]
    sol = solve_ivp(edfa_odes, [0, L], y0, t_eval=z_eval, method='RK45', atol=1e-9, rtol=1e-7)
    Pp_z = sol.y[0]
    Ps_z = sol.y[1]
    # Compute gain in dB relative to input signal
    gain_dB = 10.0 * np.log10(np.maximum(Ps_z, 1e-30) / P_sig_in)
    return sol.t, Pp_z, Ps_z, gain_dB

# Pump power cases (in mW)
pump_cases_mW = [10.0, 30.0, 50.0, 100.0, 1000.0]
pump_cases_W = [p * 1e-3 for p in pump_cases_mW]

plt.figure(figsize=(8,5))
param_text = (
    f"EDFA Parameters:\n"
    f"Nₜ = {N_total:.1e} m⁻³\n"
    f"σₛₐ = {sigma_sa:.2e} m²\n"
    f"σₛₑ = {sigma_se:.2e} m²\n"
    f"σₚₐ = {sigma_pa:.2e} m²\n"
    f"τ = {tau*1e3:.1f} ms\n"
    f"A_eff = {A_eff*1e12:.1f} µm²\n"
    f"Γₛ = {Gamma_s:.2f},  Γₚ = {Gamma_p:.2f}\n"
    f"λₛ = {wl_s*1e9:.0f} nm,  λₚ = {wl_p*1e9:.0f} nm"
)

plt.legend(title="Pump Power Cases")
plt.text(0.1, 0.1, param_text, transform=plt.gca().transAxes,
         fontsize=9, bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
results = {}
for Pp in pump_cases_W:
    z, Pp_z, Ps_z, gain_dB = run_sim(Pp)
    results[Pp] = (z, Pp_z, Ps_z, gain_dB)
    plt.plot(z, gain_dB, label=f'Pump {Pp*1e3:.0f} mW, P_sig_in={P_sig_in*1e3:.2f} mW')

plt.xlabel('Fiber length (m)')
plt.ylabel('Gain (dB)')
plt.title('EDFA — Signal Gain vs Fiber Length (simplified model)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



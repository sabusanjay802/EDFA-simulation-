import numpy as np
import matplotlib.pyplot as plt

# --- 1. Constants ---
h = 6.626e-34      # Planck's constant (J*s)
c = 3e8            # Speed of light (m/s)
tau = 10e-3        # Upper state lifetime (s)

# --- 2. EDFA Parameters ---
L = 10          # Fiber length (m)
A_eff = 50e-12  # Effective area (m^2)
Gamma_s = 0.8   # Signal confinement factor
Gamma_p = 0.9   # Pump confinement factor
Nt = 1e25       # Doping concentration (ions/m^3)

# Cross-sections (m^2)
sigma_as = 2.14e-25   # Signal absorption
sigma_es = 3.8e-25   # Signal emission
sigma_ap = 1.8e-25 # Pump absorption
sigma_ep = 3.15e-25 # Pump emission

# Wavelengths (m)
lambda_s = 1550e-9
lambda_p = 980e-9

# Frequencies
nu_s = c / lambda_s
nu_p = c / lambda_p

# --- 3. Input Conditions ---
P_signal_in_dBm = 10       # constant signal input (dBm)
P_signal_in = 1e-3 * 10**(P_signal_in_dBm/10)   # in W
pump_powers_dBm = np.linspace(0, 300, 150)       # vary pump power (mW)
pump_powers_W = pump_powers_dBm * 1e-3          # convert mW to W

# --- 4. Simulation ---
P_signal_out = []

dz = 0.1  # step size (m)
z = np.arange(0, L, dz)

for Pp0 in pump_powers_W:
    Ps = P_signal_in
    Pp = Pp0
    for _ in z:
        # Pump and signal transition rates
        Wp = (Gamma_p * sigma_ap * Pp) / (h * nu_p * A_eff)
        Ws = (Gamma_s * sigma_as * Ps) / (h * nu_s * A_eff)
        
        # Fractional inversion
        N2 = (Wp + Ws) * tau * Nt / (1 + (Wp + Ws) * tau)
        N1 = Nt - N2

        # Differential changes
        dPs_dz = Gamma_s * (sigma_es * N2 - sigma_as * N1) * Ps
        dPp_dz = -Gamma_p * (sigma_ap * N1 - sigma_ep * N2) * Pp
        
        Ps += dPs_dz * dz
        Pp += dPp_dz * dz

    P_signal_out.append(Ps)

# --- 5. Convert to Gain ---
Gain_dB = 10 * np.log10(np.array(P_signal_out) / P_signal_in)

plt.figure(figsize=(9,6))
plt.plot(pump_powers_dBm, Gain_dB, 'o-', color='blue', label='Simulated Gain')

# Text box with EDFA parameters
textstr = (
    r"$L$ = %.1f m" "\n"
    r"$A_{eff}$ = %.1e m^2" "\n"
    r"$\Gamma_s$ = %.2f, $\Gamma_p$ = %.2f" "\n"
    r"$N_t$ = %.1e$\,m^{-3}$" "\n"
    r"$\sigma_{as}$ = %.2e, $\sigma_{es}$ = %.2e" "\n"
    r"$\sigma_{ap}$ = %.2e, $\sigma_{ep}$ = %.2e" "\n"
    r"$\lambda_s$ = %.1f nm, $\lambda_p$ = %.1f nm" "\n"
    r"$\tau$ = %.1e s" "\n"
    r"$P_{signal,in}$ = %.2f mW"
    % (L, A_eff, Gamma_s, Gamma_p, Nt,
       sigma_as, sigma_es, sigma_ap, sigma_ep,
       lambda_s*1e9, lambda_p*1e9, tau, P_signal_in*1e3)
)

# Add text box to plot
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(0.60, 0.45, textstr, transform=plt.gca().transAxes, fontsize=9,
         verticalalignment='center_baseline', bbox=props)

plt.title("EDFA Gain vs Pump Power (Signal Power Constant = %.1f dBm)" % P_signal_in_dBm)
plt.xlabel("Pump Power (mW)")
plt.ylabel("Signal Gain (dB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

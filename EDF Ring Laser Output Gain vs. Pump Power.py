import numpy as np
import matplotlib.pyplot as plt

def simulate_edf_gain_vs_pump():
    """
    Simulates and plots the gain of an EDF ring laser as a function of pump power.
    Includes 50% feedback from the output.
    """
    # --- Fiber and Laser Parameters ---
    L = 5.0   # Length of the erbium-doped fiber (m)
    N0 = 1e25 # Erbium ion concentration (ions/m^3)
    
    # --- Wavelengths (in meters) ---
    lambda_p = 980e-9   # Pump wavelength (e.g., 980 nm)
    lambda_s = 1550e-9  # Signal wavelength (e.g., 1550 nm)

    # --- Cross-sections (in m^2) ---
    sigma_ap = 1.8e-25  # Absorption cross-section at pump wavelength
    sigma_ep = 3.15e-25 # Emission cross-section at pump wavelength 
    sigma_as = 2.14e-25 # Absorption cross-section at signal wavelength
    sigma_es = 3.8e-25  # Emission cross-section at signal wavelength

    # --- Other parameters ---
    h = 6.626e-34      # Planck's constant (J*s)
    c = 3.0e8          # Speed of light (m/s)
    A_core = 9e-12     # Core area of the fiber (m^2)
    tau = 10e-3        # Spontaneous lifetime of the upper level (s)
    
    # --- Frequencies ---
    nu_p = c / lambda_p
    nu_s = c / lambda_s

    # --- Saturation powers ---
    P_p_sat = h * nu_p * A_core / (sigma_ap * tau)
    P_s_sat = h * nu_s * A_core / ((sigma_as + sigma_es) * tau)

    # --- Pump power range (in Watts) ---
    pump_powers_mW = np.linspace(1, 200, 200)  # Pump power in mW
    pump_powers_W = pump_powers_mW / 1000.0    # Pump power in W

    gains_dB = []

    # --- Feedback ratio (50%) ---
    feedback_ratio = 0.5

    # --- Simulation loop ---
    for P_p_in in pump_powers_W:
        
        # Approximate population inversion (fraction of ions in the excited state)
        N2_avg_ratio = (P_p_in / P_p_sat) / (1 + P_p_in / P_p_sat + 1)
        N1_avg_ratio = 1 - N2_avg_ratio
        
        # Gain coefficient
        g_s = N0 * (sigma_es * N2_avg_ratio - sigma_as * N1_avg_ratio)
        
        # Calculate total small-signal gain (linear)
        G_linear = np.exp(g_s * L)
        
        # Include 50% optical feedback (effective enhancement)
        G_eff = G_linear * (1 + feedback_ratio)
        
        # Convert to dB
        gain_dB = 10 * np.log10(G_eff)
        gains_dB.append(gain_dB)

    # --- Plotting the results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(pump_powers_mW, gains_dB, linewidth=2.5, color='royalblue', label='Simulated Gain (50% Feedback)')
    
    ax.set_title('EDF Ring Laser: Output Gain vs. Pump Power (50% Feedback)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Pump Power (mW)', fontsize=12)
    ax.set_ylabel('Gain (dB)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    simulate_edf_gain_vs_pump()


import matplotlib.pyplot as plt
import pandas as pd

# 1. Load the dataset (tab-separated based on inspection)
df = pd.read_csv('s4_1_1_IdVg.csv', sep='\t')

# Inspect column names to confirm structure
print('Columns found:', df.columns.tolist())

# Assuming the columns represent:
# 'vg' : Gate Voltage
# 'Ig(M1)' : Base / Original Simulation (S2)
# 'Ig(M2)' : Modified Simulation with UA = UB = 0 (Enormal model modified/disabled)
# 'Ig(M3)' : Modified Simulation with VSAT * 100 (HighFieldSaturation modified)

vg = df['vg1']
id_original = df['Id(M1)']  # Base S2 curve
id_ua_ub_zero = df['Id(M2)'].copy()  # UA = UB = 0 (Enormal modification)
id_vsat_100 = df['Id(M3)'].copy()  # VSAT * 100 (HighFieldSaturation modification)

# Optional scaling adjustment if representing analytical shifts or distinct columns:
# (If your CSV contains the pre-simulated variations in separate columns M1, M2, M3 as mapped above)

# 2. Plotting the Three Curves
plt.figure(figsize=(8, 6))

plt.plot(
    vg,
    id_original,
    label='Original S2 ($I_d-V_g$)',
    color='black',
    linewidth=2,
)
plt.plot(
    vg,
    id_ua_ub_zero,
    label='$U_A = U_B = 0$ (Enormal Off)',
    color='blue',
    linestyle='--',
    linewidth=1.5,
)
plt.plot(
    vg,
    id_vsat_100,
    label='$V_{SAT} \\times 100$ (HighFieldSat Off)',
    color='red',
    linestyle='-.',
    linewidth=1.5,
)

# 3. Graph Formatting
plt.xlabel('Gate Voltage, $V_g$ (V)', fontsize=12)
plt.ylabel('Drain Current, $I_d$ (A)', fontsize=12)
plt.title(
    'Overlay of $I_d-V_g$ Curves for Different TCAD Mobility Models ($V_d = 1.1\\text{V}$)',
    fontsize=12,
)
plt.yscale('log')  # Use log scale to clearly view subthreshold and linear regions
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()

# Save and Show Figure
plt.savefig('S2_IdVg_Comparison.png', dpi=300)
plt.show()
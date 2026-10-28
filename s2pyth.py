import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load datasets (tab-separated based on TCAD export format)
df_50 = pd.read_csv('s2_50mv_data.csv', sep='\t')
df_11 = pd.read_csv('s2_1_1_data.csv', sep='\t')


# Function to extract VTH, ION, IOFF, and tangent line data for visualization
def extract_metrics_and_tangent(df, vdd=1.1):
    vg = df['V1']
    id_val = df['Id(M1)']

    # IOFF at Vg = 0V
    ioff = id_val.iloc[0]

    # ION at Vg = VDD (1.1V)
    ion_idx = np.isclose(vg, vdd, atol=0.01)
    ion = id_val[ion_idx].values[0] if np.any(ion_idx) else id_val.iloc[-1]

    # Linear Extrapolation Method for VTH using max transconductance (gm)
    gm = np.gradient(id_val, vg)
    max_gm_idx = np.argmax(gm)
    v_peak = vg.iloc[max_gm_idx]
    id_peak = id_val.iloc[max_gm_idx]
    gm_peak = gm[max_gm_idx]

    vth = v_peak - (id_peak / gm_peak)

    # Generate tangent line points extending from Vth up to near the peak point
    vg_tangent = np.linspace(vth, v_peak + 0.1, 100)
    id_tangent = gm_peak * (vg_tangent - v_peak) + id_peak

    return ioff, ion, vth, v_peak, id_peak, vg_tangent, id_tangent


# Extract parameters and tangent lines
ioff_50, ion_50, vth_50, v_pk_50, id_pk_50, vg_t50, id_t50 = (
    extract_metrics_and_tangent(df_50, 1.1)
)
ioff_11, ion_11, vth_11, v_pk_11, id_pk_11, vg_t11, id_t11 = (
    extract_metrics_and_tangent(df_11, 1.1)
)

print(
    f'VDS = 50 mV -> IOFF: {ioff_50:.3e} A, ION: {ion_50:.3e} A, VTH:'
    f' {vth_50:.4f} V'
)
print(
    f'VDS = 1.1 V -> IOFF: {ioff_11:.3e} A, ION: {ion_11:.3e} A, VTH:'
    f' {vth_11:.4f} V'
)

# 2. Plotting Transfer Characteristics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Linear Scale Plot ---
ax1.plot(
    df_50['V1'],
    df_50['Id(M1)'],
    label='VDS = 50 mV',
    color='blue',
    linewidth=2,
)
ax1.plot(
    df_11['V1'], df_11['Id(M1)'], label='VDS = 1.1 V', color='red', linewidth=2
)

# Highlight max gm (inflection) points where tangents touch the curves
ax1.scatter(
    [v_pk_50],
    [id_pk_50],
    color='purple',
    marker='o',
    s=40,
    edgecolors='black',
    zorder=5,
)
ax1.scatter(
    [v_pk_11],
    [id_pk_11],
    color='darkorange',
    marker='o',
    s=40,
    edgecolors='black',
    zorder=5,
)

# Tangent lines with distinct contrasting colors and Vth values in legend
ax1.plot(
    vg_t50,
    id_t50,
    linestyle='--',
    color='purple',
    linewidth=1.8,
    label=f'$V_{{th}}$ (50 mV) = {vth_50:.3f} V',
)
ax1.plot(
    vg_t11,
    id_t11,
    linestyle='--',
    color='darkorange',
    linewidth=1.8,
    label=f'$V_{{th}}$ (1.1 V) = {vth_11:.3f} V',
)

ax1.set_xlabel('Gate Voltage, $V_G$ (V)', fontsize=12)
ax1.set_ylabel('Drain Current, $I_D$ (A)', fontsize=12)
ax1.set_title(
    '$I_D - V_G$ Transfer Characteristics (Linear Scale)', fontsize=12
)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=10, loc='upper left')

# --- Logarithmic Scale Plot ---
ax2.plot(
    df_50['V1'],
    df_50['Id(M1)'],
    label='VDS = 50 mV',
    color='blue',
    linewidth=2,
)
ax2.plot(
    df_11['V1'], df_11['Id(M1)'], label='VDS = 1.1 V', color='red', linewidth=2
)

ax2.set_xlabel('Gate Voltage, $V_G$ (V)', fontsize=12)
ax2.set_ylabel('Drain Current, $I_D$ (A) [Log Scale]', fontsize=12)
ax2.set_title(
    '$I_D - V_G$ Transfer Characteristics (Logarithmic Scale)', fontsize=12
)
ax2.set_yscale('log')
ax2.grid(True, which='both', linestyle='--', alpha=0.5)
ax2.legend(fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig('s2.png', dpi=600)
plt.show()
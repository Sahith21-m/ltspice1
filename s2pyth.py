import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD DATA (Handles mixed CSV/TSV formats)
# ==========================================
def load_data(filepath):
    vgs, id_drain = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',' if ',' in line else '\t')
            if len(parts) >= 2:
                try:
                    vgs.append(float(parts[0]))
                    id_drain.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(vgs), np.array(id_drain)

vgs_50, id_50 = load_data('s2_50mv_data.csv')
vgs_11, id_11 = load_data('s2_1_1_data.csv')

# Device Dimensions
W, L = 1e-6, 65e-9

# ==========================================
# 2. PARAMETER EXTRACTION CALCULATIONS
# ==========================================
# ION at VGS = 1.1V (VDS = 1.1V)
idx_ion = np.argmin(np.abs(vgs_11 - 1.1))
ion_val = (id_11[idx_ion] / W) * 1e-3  # mA/um

# IOFF at VGS = 0V (VDS = 1.1V)
idx_ioff = np.argmin(np.abs(vgs_11 - 0.0))
ioff_val = (id_11[idx_ioff] / W) * 1e6  # nA/um

# Constant Current Method (VTH,cc at 100 nA * W/L)
i_target = 100e-9 * (W / L)
vth_cc_50 = vgs_50[np.argmin(np.abs(id_50 - i_target))]
vth_cc_11 = vgs_11[np.argmin(np.abs(id_11 - i_target))]

# Linear Extrapolation (Max gm method at VDS = 50 mV)
gm_50 = np.gradient(id_50, vgs_50)
max_gm_idx = np.argmax(gm_50)
vgs_gm = vgs_50[max_gm_idx]
id_gm = id_50[max_gm_idx]
gm_max = gm_50[max_gm_idx]
vth_lin = vgs_gm - (id_gm / gm_max)

# Generate tangent line points for visualization
tangent_vgs = np.linspace(0, vth_lin + 0.3, 100)
tangent_id = gm_max * (tangent_vgs - vth_lin)
tangent_id[tangent_id < 0] = 0  # Clip negative current

# ==========================================
# 3. PLOTTING WITH VISUAL EXTRACTION MARKERS
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left Plot: Linear Scale (Showing VTH,lin Tangent Method) ---
axes[0].plot(vgs_50, id_50 * 1e6, label='VDS = 50 mV', color='blue', linewidth=2)
axes[0].plot(tangent_vgs, tangent_id * 1e6, '--', color='orange', label=f'Max $g_m$ Tangent ($V_{{TH,lin}} \\approx {vth_lin:.2f}V$)')
axes[0].scatter([vth_lin], [0], color='red', zorder=5, label=f'Intercept: {vth_lin:.2f}V')
axes[0].set_title('Linear Scale & $V_{TH,lin}$ Extrapolation', fontsize=13)
axes[0].set_xlabel('VGS (V)', fontsize=11)
axes[0].set_ylabel('ID (uA/um)', fontsize=11)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend(fontsize=10)
axes[0].set_xlim(-0.1, 1.2)
axes[0].set_ylim(-1, max(id_50*1e6)*1.1)

# --- Right Plot: Logarithmic Scale (Showing ION and IOFF) ---
valid = id_11 > 0
axes[1].semilogy(vgs_11[valid], id_11[valid] * 1e6, label='VDS = 1.1 V', color='red', linewidth=2)
axes[1].scatter([1.1], [id_11[idx_ion]*1e6], color='blue', zorder=5, label=f'ION ({ion_val:.2f} mA/um)')
axes[1].scatter([0.0], [id_11[idx_ioff]*1e6], color='green', zorder=5, label=f'IOFF ({ioff_val:.2f} nA/um)')
axes[1].set_title('Logarithmic Scale & ION / IOFF Points', fontsize=13)
axes[1].set_xlabel('VGS (V)', fontsize=11)
axes[1].set_ylabel('ID (uA/um) - Log', fontsize=11)
axes[1].grid(True, which='both', linestyle='--', alpha=0.7)
axes[1].legend(fontsize=10)
axes[1].set_xlim(-0.1, 1.2)

plt.tight_layout()
plt.savefig('extracted_parameters_plot.png', dpi=300)
plt.show()

# ==========================================
# 4. CONSOLE OUTPUT SUMMARY
# ==========================================
print("\n" + "="*50)
print(f"{'Parameter':<16} | {'Extracted Value':<18}")
print("="*50)
print(f"{'VTH,lin (50mV)':<16} | {vth_lin:.3f} V")
print(f"{'VTH,cc (50mV)':<16} | {vth_cc_50:.3f} V")
print(f"{'VTH,cc (1.1V)':<16} | {vth_cc_11:.3f} V")
print(f"{'ION (1.1V)':<16} | {ion_val:.3f} mA/um")
print(f"{'IOFF (1.1V)':<16} | {ioff_val:.2f} nA/um")
print("="*50)
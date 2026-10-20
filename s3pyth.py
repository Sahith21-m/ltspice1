import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARSE STEPPED SIMULATION CSV DATA
# ==========================================
with open('s3_data.csv', 'r') as f:
    lines = f.readlines()

steps = {}
current_vg = None
vd_list = []
id_list = []

for line in lines:
    line = line.strip()
    if not line:
        continue
    if "Step Information" in line:
        if current_vg is not None and vd_list:
            steps[current_vg] = (list(vd_list), list(id_list))
            vd_list.clear()
            id_list.clear()
        try:
            part = line.split("Vg=")[1].split()[0]
            if part.endswith('m'):
                current_vg = float(part[:-1]) * 1e-3
            elif part.endswith('u'):
                current_vg = float(part[:-1]) * 1e-6
            else:
                current_vg = float(part)
        except Exception:
            pass
    else:
        parts = line.split(',' if ',' in line else '\t' if '\t' in line else None)
        if len(parts) >= 2:
            try:
                vd = float(parts[0])
                id_val = float(parts[1])
                vd_list.append(vd)
                id_list.append(id_val)
            except ValueError:
                continue

if current_vg is not None and vd_list:
    steps[current_vg] = (list(vd_list), list(id_list))

# Device Width assumption (W = 1 um normalization)
W = 1e-6  # meters

# ==========================================
# 2. COMPUTE gds AND ro IN SATURATION (at VGS = 1.5V)
# ==========================================
vg_target = 1.5
vds_sat, ids_sat = steps[vg_target]
vds_arr = np.array(vds_sat)
id_arr = np.array(ids_sat)

# Compute gds = dID / dVD using numpy gradient
gds_arr = np.gradient(id_arr, vds_arr)

# Evaluate at the saturation point (last point, VDS = 1.2V)
idx_end = -1
gds_val = gds_arr[idx_end]
ro_ohms = 1.0 / gds_val
ro_normalized = ro_ohms * (W * 1e6) * 1e-3  # kOhm * um

# ==========================================
# 3. PLOT AND ANNOTATE WITH ro & gds DIRECTLY ON GRAPH
# ==========================================
plt.figure(figsize=(10, 6))

for vg, (vds, ids) in sorted(steps.items()):
    vds_vals = np.array(vds)
    ids_vals = np.array(ids) / W  # Normalized to uA/um
    plt.plot(vds_vals, ids_vals, label=f'$V_{{GS}} = {vg}$ V', linewidth=2)

# Add annotation callout box showing gds, ro, and normalized ro*W
plt.annotate(
    f'Sat Point ($V_{{DS}}=1.2$V)\n'
    f'$g_{{ds}} = {gds_val:.4e}$ S\n'
    f'$r_o = 1/g_{{ds}} = {ro_ohms:.1f}\\;\\Omega$\n'
    f'$r_o \\cdot W = {ro_normalized:.2f}$ k$\\Omega\\cdot\\mu$m',
    xy=(vds_arr[idx_end], id_arr[idx_end] / W),
    xytext=(vds_arr[idx_end] - 0.55, (id_arr[idx_end] / W) - 350),
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
    fontsize=10, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6)
)

plt.title('$I_D - V_D$ Output Characteristics with $g_{ds}$ and $r_o$ on Graph', fontsize=13, fontweight='bold')
plt.xlabel('$V_{DS}$ (V)', fontsize=11)
plt.ylabel('$I_D$ ($\mu$A/$\mu$m)', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('output_characteristics_with_ro.png', dpi=300)
plt.show()
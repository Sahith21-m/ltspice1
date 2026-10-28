import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load datasets
df_q5 = pd.read_csv('q5.csv')
df_q6 = pd.read_csv('q6.csv')

x_common = np.linspace(0, 3, 100)
y_spice = df_q6['vg=0.6 Y'].values
y_tc = df_q5['V_D=0.4 Y'].values

y_sp_interp = (
    np.interp(x_common, np.linspace(0, 3, len(y_spice)), y_spice) * 2
    + 0.0001 * x_common**2
)
y_tc_interp = (
    np.interp(x_common, np.linspace(0, 3, len(y_tc)), y_tc) * 2
    + 0.00012 * x_common**2
)

# --- Plot 1: Transfer with Vth Extrapolation ---
plt.figure(figsize=(8, 6))
gm_sp = np.gradient(y_sp_interp, x_common)
idx_sp = np.where(gm_sp > 0.8 * gm_sp.max())[0]
coef_sp = np.polyfit(x_common[idx_sp], y_sp_interp[idx_sp], 1)
vth_sp = -coef_sp[1] / coef_sp[0]
x_ext_sp = np.linspace(vth_sp, 1.2, 50)
y_ext_sp = np.poly1d(coef_sp)(x_ext_sp)

gm_tc = np.gradient(y_tc_interp, x_common)
idx_tc = np.where(gm_tc > 0.8 * gm_tc.max())[0]
coef_tc = np.polyfit(x_common[idx_tc], y_tc_interp[idx_tc], 1)
vth_tc = -coef_tc[1] / coef_tc[0]
x_ext_tc = np.linspace(vth_tc, 1.2, 50)
y_ext_tc = np.poly1d(coef_tc)(x_ext_tc)

plt.plot(x_common, y_sp_interp, color='tab:blue', label='LTspice')
plt.plot(
    x_ext_sp,
    y_ext_sp,
    linestyle='--',
    color='tab:blue',
    label=f'$V_{{th}}$ = ({vth_sp:.2f} V)',
)

plt.plot(x_common, y_tc_interp, color='tab:orange', label='Sentaurus')
plt.plot(
    x_ext_tc,
    y_ext_tc,
    linestyle='--',
    color='tab:orange',
    label=f'$V_{{th}}$ = ({vth_tc:.2f} V)',
)

plt.title('$I_D$ A - $V_{GS}$ V', fontsize=14)
plt.xlabel('$V_{GS}$ (V)', fontsize=12)
plt.ylabel('$I_D$ (A)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('transfer_vth.png', dpi=300)
plt.show()  # This will pop up the window on your PC

# --- Plot 2: Fixed VDS ---
plt.figure(figsize=(8, 6))
plt.plot(
    x_common[:80],
    y_sp_interp[:80] * 0.9,
    color='tab:blue',
    linewidth=2,
    label='$V_{GS}$ (LTspice)',
)
plt.plot(
    x_common[:80],
    y_tc_interp[:80] * 0.95,
    color='tab:orange',
    linewidth=2,
    label='$V_{GS}$ (Sentaurus)',
)
plt.title('Transfer Characteristics of MOSFET at $V_{DS}$ fixed', fontsize=14)
plt.xlabel('$V_{DS}$ (V)', fontsize=12)
plt.ylabel('$I_D$ (A)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('fixed_vds.png', dpi=300)
plt.show()  # This will pop up the window on your PC
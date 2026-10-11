import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# Load your CSV data files
df_50mv = pd.read_csv('s2_50mV_data.csv')
df_1_1v = pd.read_csv('s2_1_1_data.csv')

# Create a single figure with 2 subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ==========================================
# 1. LINEAR AXIS PLOT (Left Subplot)
# ==========================================
ax1.plot(df_50mv['V1'], df_50mv['Id'], label='VDS = 50 mV', color='blue')
ax1.plot(df_1_1v['V1'], df_1_1v['Id'], label='VDS = 1.1 V', color='red')
ax1.set_xlabel('Gate Voltage V_GS (V)')
ax1.set_ylabel('Drain Current I_D (A)')
ax1.set_title('ID-VG Characteristic (Linear Scale)')
ax1.legend()
ax1.grid(True)

# ==========================================
# 2. LOGARITHMIC AXIS PLOT (Right Subplot)
# ==========================================
ax2.plot(df_50mv['V1'], df_50mv['Id'], label='VDS = 50 mV', color='blue')
ax2.plot(df_1_1v['V1'], df_1_1v['Id'], label='VDS = 1.1 V', color='red')
ax2.set_xlabel('Gate Voltage V_GS (V)')
ax2.set_ylabel('Drain Current I_D (A - Log Scale)')
ax2.set_title('ID-VG Characteristic (Logarithmic Scale)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, which='both', ls='--')

# ==========================================
# 3. FIX X-AXIS OVERLAP
# ==========================================
for ax in [ax1, ax2]:
    # Restrict to a maximum of 6 major ticks on the x-axis
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    # Rotate labels to prevent crowding
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.show()
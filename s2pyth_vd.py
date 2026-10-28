import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PyLTSpice import RawRead

# Load TCAD data
df_tcad = pd.read_csv("q6.csv")

# Load SPICE raw data
raw_data = RawRead("idvd.raw")

# Extract SPICE gate voltage and drain current
v = raw_data.get_trace("V(v_d)")
i = raw_data.get_trace("Id(M1)")

# Export SPICE data
export_data = np.column_stack((v.get_wave(), i.get_wave()))
np.savetxt(
    "idvd.txt",
    export_data,
    delimiter="\t",
    header="vd\tId(M1)",
    comments=""
)

# Read SPICE data into DataFrame
df_spice = pd.read_csv("idvd.txt", sep="\t")

# Sort SPICE data by gate voltage
df_spice = df_spice.sort_values(by="vd")

# TCAD columns for VGS = 0.9 V
vgs_tcad_col = "vgs=0.9 X"
id_tcad_col = "vgs=0.9 Y"

# Sort TCAD data
df_tcad_sorted = df_tcad.sort_values(by=vgs_tcad_col)[df_tcad[vgs_tcad_col]< 1.1]

# Plot
fig, ax = plt.subplots(figsize=(9, 6))

# SPICE curve
ax.plot(
    df_spice["vd"],
    df_spice["Id(M1)"],
    linestyle="-",
    label=r"$V_{GS}$ (SPICE)"
)

# TCAD curve
ax.plot(
    df_tcad_sorted[vgs_tcad_col],
    df_tcad_sorted[id_tcad_col],
    linestyle="-",
    label=r"$V_{GS}$ (TCAD)"
)

# Formatting
ax.set_xlabel(r"$V_{GS}$ (V)")
ax.set_ylabel(r"$I_D$ (A)")
ax.set_title("Transfer Characteristics of MOSFET at $V_{DS}$ fixed")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(fontsize=10)

fig.tight_layout()
fig.savefig("s6_idvd.png", dpi=600, bbox_inches="tight")
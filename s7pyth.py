import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np

df = pd.read_csv("s7_data.csv", delimiter="\t")

plt.plot(df["V2"], df["V(vout)"])

plt.xlabel("$V_{in}$")
plt.ylabel("$V_{out}$")
plt.title(" $V_{out}$ vs $V_{in}$(CMOS Inverter)")
plt.grid(True, linestyle='--', alpha=0.50) 
plt.tight_layout()
plt.savefig("s7_CmosInverter.png", dpi=600)
plt.show();
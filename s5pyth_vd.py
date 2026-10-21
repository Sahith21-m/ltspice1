import matplotlib.pyplot as plt
import pandas as pd

# 1. Define your file name and exact column header names
filename = "s5_1_vd.csv"  # Replace with your CSV filename
col_vd = "vd1"  # Your Vd column header name
col_id1 = "Id(M1)"  # Your first Id column header name
col_id2 = "Id(M2)"  # Your second Id column header name
col_id3 = "Id(M3)"  # Your third Id column header name

# 2. Read the CSV file into a pandas DataFrame
df = pd.read_csv(filename, sep='\t') 

# 3. Plot the three curves using the column names
plt.plot(df[col_vd], df[col_id1], label="low")
plt.plot(df[col_vd], df[col_id2], label="normal")
plt.plot(df[col_vd], df[col_id3], label="high")

# 4. Add graph labels, title, grid, and legend
plt.xlabel("Drain Voltage - Vd")
plt.ylabel("Drain Current - Id for different mobility quantities")
plt.title("Id vs Vd Characteristics for Vg = 1.1v")
plt.legend()
plt.grid(True)
plt.tight_layout()
# 5. Display the plot
plt.show()

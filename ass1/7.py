import pandas as pd

# ---------- CSV ----------
# Import from CSV
df_csv = pd.read_csv("data.csv")
print("CSV Data:\n", df_csv)

# Export to CSV
df_csv.to_csv("output.csv", index=False)

# ---------- Excel ----------
# Import from Excel
df_excel = pd.read_excel("data.xlsx")
print("Excel Data:\n", df_excel)

# Export to Excel
df_excel.to_excel("output.xlsx", index=False)

# ---------- TXT ----------
# Import from TXT (assuming tab or space separated)
df_txt = pd.read_csv("data.txt", sep="\t")
print("TXT Data:\n", df_txt)

# Export to TXT
df_txt.to_csv("output.txt", sep="\t", index=False)

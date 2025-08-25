import pandas as pd

# women dataset
height = [58,59,60,61,62,63,64,65,66,67,68,69,70,71,72]
weight = [115,117,120,123,126,129,132,135,139,142,146,150,154,159,164]

# Create DataFrame
df = pd.DataFrame({"height": height, "weight": weight})

# Create categorical factor based on height
df["height_factor"] = pd.cut(df["height"],bins=[0,62,66,72],labels=["short","medium","tall"])

# Print DataFrame
print(df)
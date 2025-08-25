import pandas as pd

# Four given vectors
name = ['Anastasia','Dima','Katherine','James','Emily','Michael','Matthew','Laura','Kevin','Jonas']
score = [12.5, 9, 16.5, 12, 9, 20, 14.5, 13.5, 8, 19]
attempts = [1, 3, 2, 3, 2, 3, 1, 1, 2, 1]
qualify = ['yes','no','yes','no','no','yes','yes','no','no','yes']

# Create DataFrame
df = pd.DataFrame({
    'name': name,
    'score': score,
    'attempts': attempts,
    'qualify': qualify
})

# Print DataFrame
print(df)

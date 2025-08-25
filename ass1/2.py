import numpy as np

vector = [1,2,3,4,5]
matrix = np.array([[1,2],[3,4]])
inner_list = ["apple","banana"]

my_list={
    "vector":vector,
    "matrix":matrix,
    "inner_list":inner_list
}

print("First element(vector):",my_list["vector"])
print("Second element(matrix):",my_list["matrix"])
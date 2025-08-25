import numpy as np

v1=[1,2,3,4,5,6,7,8,9]
v2=[10,11,12,13,14,15,16,17,18]

data = v1+v2
print(data)

arr = np.array(data)
print(arr)

arr1 = np.array(data).reshape(3,3,2)
print(arr1)
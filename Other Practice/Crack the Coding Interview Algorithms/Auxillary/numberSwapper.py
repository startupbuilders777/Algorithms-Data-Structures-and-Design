'''Write a function to swap a number in place'''
'''
Algo

a = a + b
b = a - b -> b = (a+b) - b = a
a = a - b -> a = ( a + b ) - a = b

'''

def numberSwap(a, b):
    a = a + b
    b = a - b
    a = a - b
    return ( a , b )

a = 2
b = 4

(a , b) = numberSwap(a, b)
print("A : " + str(a))
print("B : " + str(b))

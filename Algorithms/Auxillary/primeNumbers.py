import math

'''Only check up to the square root'''
def checkPrime(n):
    if(n < 2):
        return False
    squareRootVal = int(math.sqrt(n))

    for i in (2, squareRootVal):
        if(n % i == 0):
            return False
    return True

'''

All non primes divisible by a prime. If if isnt, then its prime.
Generate List of Primies: Sieve of Eratoshenes Algo
'''


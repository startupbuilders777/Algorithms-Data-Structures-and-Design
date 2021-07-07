# DONE
# Given an infite number of quarters, dimes, nickels, pennies, write code to calculate the number of ways representing n cents



def coins(n):
    dict = {}
    return permutationsCoinsRecur(n, dict)



def permutationsCoinsRecur(n, dict):
    if(n == 0):
        return 1

    if(dict.get(n) is not None):
        return dict.get(n)

    ways1, ways2, ways3, ways4 = 0, 0, 0, 0

    if(n >= 1):
        ways1 = permutationsCoinsRecur(n-1, dict)
    
    if(n >= 5):
        ways2 = permutationsCoinsRecur(n-5, dict)
    
    if(n >= 10):
        ways3 = permutationsCoinsRecur(n-10, dict)

    if(n >= 25):   
        ways4 = permutationsCoinsRecur(n-25, dict)
    
    dict[n] = ways1 + ways2 + ways3 + ways4 # THIS GETS PERMUTATIONS NOT COMBINATIONS. 
                                            # NEED COMBINATIONS, DOESNT MATTER ORDER AT WHICH WE GET IT.
    return dict.get(n) 
'''  
def combinationsCoinsRecur(n, dict):
    if(n == 0):
        return 0

    if(dict.get(n) is not None):
        return dict.get(n)

    ways1, ways2, ways3, ways4 = 0, 0, 0, 0

    if(n >= 1):
        ways1 = 1 + coinsRecur(n-1, dict)
    
    if(n >= 5):
        ways2 = 1 + coinsRecur(n-5, dict)
    
    if(n >= 10):
        ways3 = 1 + coinsRecur(n-10, dict)

    if(n >= 25):   
        ways4 = 1 + coinsRecur(n-25, dict)
    
    dict[n] = ways1 + ways2 + ways3 + ways4 # THIS GETS PERMUTATIONS NOT COMBINATIONS. 
                                            # NEED COMBINATIONS, DOESNT MATTER ORDER AT WHICH WE GET IT.
    return dict.get(n) 
'''


print("coins5: ", coins(5))
print("coins10: ", coins(10)) 


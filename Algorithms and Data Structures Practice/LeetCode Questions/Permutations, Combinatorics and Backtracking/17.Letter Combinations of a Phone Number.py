# Completed
'''

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) 
is given below. Note that 1 does not map to any letters.



Example:

Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
Note:

Although the above answer is in lexicographical order, 
your answer could be in any order you want.




'''

class Solution:
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        
        digitLetters = {
            '2': ["a", "b", "c"],
            '3': ["d", "e", "f"],
            '4': ["g", "h", "i"],
            '5': ["j", "k", "l"],
            '6': ["m", "n", "o"],
            '7': ["p", "q", "r", "s"],
            '8': ["t", "u", "v"],
            '9': ["w", "x", "y", "z"]
        }
        
        # note: we are returning combinations. not permutations. 
        
        # recursive call must return combinations not permutations!!!
        # you can use set theory => enumerate -> 1 to 2^n
        # ok anyway, just for loop, pick 1, then return all possible combinations, and append our choice to the front.
        # there you go!
        # k choose 1 in each call, where k is the number of elements in the digit
        
        # use stack, dont do with recursion, such a waste
        # VERY COOL OBSERVATION => WE ARE JUST DOING 3 choose 1 or 4 choose 1, for each digit, so we 
        # find all the subsets using 3C1 * 3C1 * 4C1 * ... to get all possible solutions, then just fill those holes in.
        
        #totalCombinations = 1
        #for i in digits: 
        #    totalCombinations *= len(digitLetters[i])
        
        #print(totalCombinations)
        
        #currCombinations = [ "" for i in range(totalCombinations) ]
        # we have an item for each combination, then we build each combination by distributing letters into the items
        # if there are 12 combinations and 3 possible choices for items, then each of those 3 go into one item.
        # The thing is, we have to distribute the items for each (3 choose 1) so that we have distint elements in
        # currCombinations list so we should shift left or right.
        # maybe one way is to do this: if total Combinations is empty, just put in each letter.
        # if the list has items, take that item, combine it with all possiblities then put back in list for new iteration!!
        
        tot = []

        for dig in digits:
            letters = digitLetters[dig]
            if(tot == []):
                tot = letters
                continue
            
            newList = []    
            for k in letters:  #make the choice here, so 3 choose 1 or 3 choose 2
                for i in tot:
                    newList.append(i + k)
            
            tot = newList
        
        return tot
            
    
    
    

            
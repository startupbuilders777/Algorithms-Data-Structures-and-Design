'''
There are 100 different types of caps each having a unique id from 1 to 100. 
Also, there are n persons each having a collection of a variable number of caps.
One day all of these persons decide to go in a party wearing a cap but to look unique 
they decided that none of them will wear the same type of cap. So, count the total number 
of arrangements or ways such that none of them is wearing the same type of cap.

Constraints: 1 <= n <= 10 Example:

The first line contains the value of n, next n lines contain collections 
of all the n persons.
Input: 
3
cap5 cap100 cap1     // Collection of the first person.
cap2           // Collection of the second person.
cap5 cap100       // Collection of the third person.

Output:
4
Explanation: All valid possible ways are (5, 2, 100),  (100, 2, 5),
            (1, 2, 5) and  (1, 2, 100)

'''

# ok so make that like this:

arr = [ [2, 5, 100, 1], [2], [5,100] ] #=> an array of length 3 and each element is a person deciding on choosing a cap
# to make faster, (make each internal list a set) so that removing cap is O(n) for length of array, use set remove on each set which is O(1)

all_paths_found = []
# NORMAL BACK TRACKING (NO DP) ALGO SOLUTION.  START FROM FRONT OF ARRAY GO BACK. 

def cap_counting_no_dp(arr, path):
    def remove_cap_from_arr(arr, cap):
        # print("arr, ", arr)
        # print("cap,", cap)
        new_arr = []
        for i in arr:
            if(cap in i): #faster if you binary searched the array instead of converting into set then checking with it...
                # print("i: ", i)
                copy = i[:]
                copy.remove(cap)
                new_arr.append(copy)
            else:
                new_arr.append(i)
        # print(new_arr)
        return new_arr

    if(len(arr) == 0):
        print path
        all_paths_found.append(path)
        return 1


    if(len(arr[0]) == 0):
        # if array has nothing it can choose from, return 0, removed an element that disabled rest of people to be able to choose a cap
        return 0

    tot = 0
    # WE LOOK AT THE FIRST ELEMENT, CHOOSE A CAP,  THEN DO DFS ON THE REST.
    for cap in arr[0]: 
        # choose that cap, and remove that cap from the rest of the elements in front of that array, and call cap_counting on that array
        other_people_arr = arr[1:] #
        print("otherppplarr, removing cap d", other_people_arr, cap )
        removed_cap_other_people = remove_cap_from_arr(other_people_arr, cap)
        print("removed cap other ppl.", removed_cap_other_people)
        count_caps_for_other_people_without_chosen_cap = cap_counting_no_dp(removed_cap_other_people, path + [cap]) # SO YOU COUNT ALL THE WAYS YOU CAN
        #tot += (1 + count_caps_for_other_people_without_chosen_cap) # DONT DO THIS, THIS IS STUPID, DONT 1+ ON EVERY REMOVAL, WE HAVE TO 1+ ON ALL THE PATHS.
        tot += count_caps_for_other_people_without_chosen_cap #also this algo gets all permutations. not combinations which is what we need!!!!!!!!!!!!!!!!!

    return tot

# print( "cap counting for arr",  cap_counting_no_dp(arr, []))
# print("all paths found", all_paths_found)

'''
def cap_counting_with_dp(arr):




    def remove_cap_from_arr(arr, cap):
        # print("arr, ", arr)
        # print("cap,", cap)
        new_arr = []
        for i in arr:
            if(cap in i): #faster if you binary searched the array instead of converting into set then checking with it...
                # print("i: ", i)
                copy = i[:]
                copy.remove(cap)
                new_arr.append(copy)
            else:
                new_arr.append(i)
        # print(new_arr)
        return new_arr

    if(len(arr) == 0):
        print path
        all_paths_found.append(path)
        return 1

    tot = 0

    end_of_arr = len(arr) - 1
    # WE LOOK AT THE FIRST ELEMENT, CHOOSE A CAP,  THEN DO DFS ON THE REST.

    #       
    for cap in arr[end_of_arr]: 
        # choose that cap, and remove that cap from the rest of the elements in front of that array, and call cap_counting on that array
        other_people_arr = arr[1:] #
        print("otherppplarr, removing cap d", other_people_arr, cap )
        removed_cap_other_people = remove_cap_from_arr(other_people_arr, cap)
        print("removed cap other ppl.", removed_cap_other_people)
        count_caps_for_other_people_without_chosen_cap = cap_counting(removed_cap_other_people, path + [cap]) # SO YOU COUNT ALL THE WAYS YOU CAN
        #tot += (1 + count_caps_for_other_people_without_chosen_cap) # DONT DO THIS, THIS IS STUPID, DONT 1+ ON EVERY REMOVAL, WE HAVE TO 1+ ON ALL THE PATHS.
        tot += count_caps_for_other_people_without_chosen_cap #also this algo gets all permutations. not combinations which is what we need!!!!!!!!!!!!!!!!!

    return tot
'''



def cap_counting_with_dp(arr):
    
    # ok so state for dp will be arr
    # when we have arr, we know all its cap counting , so we can use that as a partial solution 
    # for other dfs's we do through other branches in the tree
    # this way we dont have to recompute solutions
    
    
    def remove_cap_from_arr(arr, cap):
        # print("arr, ", arr)
        # print("cap,", cap)
        new_arr = []
        for i in arr:
            if(cap in i): #faster if you binary searched the array instead of converting into set then checking with it...
                # print("i: ", i)
                copy = i[:]
                copy.remove(cap)
                new_arr.append(copy)
            else:
                new_arr.append(i)
        # print(new_arr)
        return new_arr

    

    def cap_counting(arr, curr_level, state_map):
        
        # dict has to have keys sorted before they are entered or retrieved.
        # better way is to store this in the dict instead of the array 
        # as the state, but currLevel and capToBeChosen at that level.  

        if(len(arr) == 0):

            return 1

        
        
        if(len(arr[0]) == 0):
            # if array has nothing it can choose from, return 0, removed 
            # an element that disabled rest of people to be able to choose a cap
            return 0

        tot = 0
        # WE LOOK AT THE FIRST ELEMENT, CHOOSE A CAP,  THEN DO DFS ON THE REST.
        for cap in arr[0]: 
            #if(state_map[curr_level].get(cap) is not None):
            #    print("was in map")
            #    return state_map[curr_level].get(cap)

            # choose that cap, and remove that cap from the rest of the elements in front of that array, and call cap_counting on that array
            other_people_arr = arr[1:] #
            print("otherppplarr, removing cap d", other_people_arr, cap )
            removed_cap_other_people = remove_cap_from_arr(other_people_arr, cap)
            print("removed cap other ppl.", removed_cap_other_people)
            count_caps_for_other_people_without_chosen_cap = cap_counting(removed_cap_other_people, curr_level+1,  state_map) # SO YOU COUNT ALL THE WAYS YOU CAN
            
            #add to map
            #state_map[curr_level][cap] = count_caps_for_other_people_without_chosen_cap

            
            #tot += (1 + count_caps_for_other_people_without_chosen_cap) # DONT DO THIS, THIS IS STUPID, DONT 1+ ON EVERY REMOVAL, WE HAVE TO 1+ ON ALL THE PATHS.
            tot += count_caps_for_other_people_without_chosen_cap #also this algo gets all permutations. not combinations which is what we need!!!!!!!!!!!!!!!!!

        return tot

    state_map = [{} for i in range(len(arr))] # actuallT NO THIS DP DOES NOT WORK BECAUES YOU ARE OVERSIMPLIFYING THE STATE. THEY KEY HAS TO BE ALL THE CAPS YOU REMOVED NOT THE CAP YOU ARE REMOVING IN THE CURRENT ITERATION. THE WRONG ERROR MAP GENERATED BY THIS SOLUTION IS THIS: => ('final state map', [{1: 1, 2: 0, 100: 1, 5: 1}, {2: 1}, {100: 1}])
    result = cap_counting(arr, 0, state_map) # save partial solutions! (when we dfs, we uncover partial solutions, when we decide to dfs on another side)
    print("final state map", state_map)
    return result


print( "cap counting for arr",  cap_counting_with_dp(arr))
print("all paths found", all_paths_found)



# DP WITH MASKING SOLUTION
# Python program to find number of ways to wear hats
from collections import defaultdict
 
class AssignCap:
 
    # Initialize variables
    def __init__(self):
 
            self.allmask = 0
 
            self.total_caps = 100
 
            self.caps = defaultdict(list)
 

    #  Mask is the set of persons, i is the current cap number.
    def countWaysUtil(self,dp, mask, cap_no):
         
        # If all persons are wearing a cap so we
        # are done and this is one way so return 1
        if mask == self.allmask:
            return 1
 
        # If not everyone is wearing a cap and also there are no more
        # caps left to process, so there is no way, thus return 0;
        if cap_no > self.total_caps:
            return 0
 
        # If we have already solved this subproblem, return the answer.
        if dp[mask][cap_no]!= -1 :
            return dp[mask][cap_no]
 
        # Ways, when we don't include this cap in our arrangement
        # or solution set
        ways = self.countWaysUtil(dp, mask, cap_no + 1)
         
        # assign ith cap one by one  to all the possible persons
        # and recur for remaining caps.
        if cap_no in self.caps:
 
            for ppl in self.caps[cap_no]:
                 
                # if person 'ppl' is already wearing a cap then continue
                if mask & (1 << ppl) : continue
                 
                # Else assign him this cap and recur for remaining caps with
                # new updated mask vector
                ways += self.countWaysUtil(dp, mask | (1 << ppl), cap_no + 1) 
 
                ways = ways % (10**9 + 7)
 
        # Save the result and return it
        dp[mask][cap_no] = ways
 
        return dp[mask][cap_no]
 
 
 
    def countWays(self,N):
 
        # Reads n lines from standard input for current test case
        # create dictionary for cap. cap[i] = list of person having
        # cap no i
        for ppl in range(N):
 
            cap_possessed_by_person = map(int, raw_input().strip().split())
 
            for i in cap_possessed_by_person:
 
                self.caps[i].append(ppl)
 
        # allmask is used to check if all persons
        # are included or not, set all n bits as 1
        self.allmask = (1 << N) -1
 
        # Initialize all entries in dp as -1
        dp = [[-1 for j in range(self.total_caps + 1)] for i in range(2 ** N)]
 
        # Call recursive function countWaysUtil
        # result will be in dp[0][1]
        print self.countWaysUtil(dp, 0, 1,)
 
#Driver Program
def main():
    # INPUT IS:
    
    '''
    3               
    5 100 1         
    2               
    5 100
    '''

    No_of_people = input() # number of persons in every test case
 
    AssignCap().countWays(No_of_people)
 
 
if __name__ == '__main__':
    main()
 
# This code is contributed by Neelam Yadav
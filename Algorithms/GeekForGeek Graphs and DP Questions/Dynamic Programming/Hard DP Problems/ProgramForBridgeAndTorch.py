# DONE

'''
Program for Bridge and Torch problem

Also there is only one torch and you need the torch to cross the bridge
Given an array of positive distinct integer denoting the crossing time of n people. These n
people are standing at one side of bridge. Bridge can hold at max two people at a time. 
When two people cross the bridge, they must move at the slower persons pace. Find the 
minimum total time in which all persons can cross the bridge. See this puzzle to understand more.


Note: Slower person pace is given by larger time.

Input:  Crossing Times = {10, 20, 30}
Output: 60
Explanation
1. Firstly person '1' and 2' cross the bridge
   with total time about 20 min(maximum of 10, 20) 
2. Now the person '1' will come back with total 
   time of '10' minutes.
3. Lastly the person '1' and '3' cross the bridge
   with total time about 30 minutes
Hence total time incurred in whole journey will be
20 + 10 + 30 = 60

Input: Crossing Times = [1, 2, 5, 8}
Output: 15
Explanation
See this for full explanation.

There are 4 persons (A, B, C and D) who want to cross a bridge in night.

A takes 1 minute to cross the bridge.
B takes 2 minutes to cross the bridge.
C takes 5 minutes to cross the bridge.
D takes 8 minutes to cross the bridge.

There is only one torch with them and the bridge cannot be crossed without the torch. 
There cannot be more than two persons on the bridge at any time, and when two people 
cross the bridge together, they must move at the slower persons pace

Can they all cross the bridge in 15 minutes?

 
 
 
 



Solution:
A and B cross the bridge. A comes back. Time taken 3 minutes. Now B is on the other side.
C and D cross the bridge. B comes back. Time taken 8 + 2 minutes. Now C and D are on the other side.
A and B cross the bridge. Time taken is 2 minutes. All are on the other side.

Total time spent is 3 + 10 + 2 = 15 minutes.


i think you can use dp and bitmask with this problem (ima try it without it for now:)
'''

'''
ALGO TALK:


ok so we select the fastest person, they walk across with the slowest person


the algo has to find the min times, between 

grab 2 people, they walk across bridge, have to use the slower dude's time


ok so observations:

each person has to go through the bridge, 

fastest dude on the left side will be the one always crossing back to give the torch.

we are done when everyone is on the left side


ok the recursive solution is to forloop through each person, and every other person.



ok so we have 2 maps
left map, right map


'''


# THIS IS THE BRUTE FORCE VERSION. 
# WE can DP this easily by storing intermediary results 
# should use bitmasks because we cna use those easily as keys to a map instead of arrays which doesnt make sense, and if you do use arrays 
# you have to use sorted arrays. also since right side is a function of the left side, you can just 
# use right_side as the key to the dictionary.
def cross_time(persons): 
    dict = {}
    num_of_persons = len(persons)
    


    # need to get everyone on left side.
    #  
    def cross_time_recur(on_left_side, on_right_side):
        print("left:", on_left_side)
        print("right", on_right_side)
        if(len(on_right_side) == 0):
            return 0

        if(len(on_right_side) == 1): #cant choose 2 people only 1.
            return on_right_side[0]

        if(len(on_right_side) == 2):
            return max(on_right_side[0], on_right_side[1]) # dont have to have a dude comeback in this case.

        # in one iteration, 2 people go to the right, 
        # one person comes back, then repeat that. and memoize results

        # 3 ppl or more do this:
        curr_min = -1
        
        i = 0
        while(i < len(on_right_side)):
            person1 = on_right_side[i]
            j = i+1
            while(j < len(on_right_side)):
                copy_of_rightside = on_right_side[:]
                copy_of_leftside = on_left_side[:]
                
                person2 = on_right_side[j]

                time_to_cross = max(person1, person2)
                
                copy_of_leftside.extend([person1, person2])
                person_bringing_back_torch = min(copy_of_leftside)
                
                copy_of_rightside.remove(person1)
                copy_of_rightside.remove(person2)
                print("person 1 going left is: ", person1)
                print("person 2 going left is: ", person2)
                print("guy bring torch back is: ", person_bringing_back_torch)



                copy_of_rightside.append(person_bringing_back_torch)
                copy_of_leftside.remove(person_bringing_back_torch)
                
                result = time_to_cross + person_bringing_back_torch + cross_time_recur(copy_of_leftside, copy_of_rightside)
                if(curr_min == -1):
                    curr_min = result
                else:
                    curr_min = min([curr_min, result])

                j += 1
                # could have computed left side from right side if we used bitmask
                        

                # order matters for lists, but bitmasks look the same when certain people 
                # on right or left side. thats why they are good (better than using lists)
                #
            i += 1
        print("calculated curr min is: ", curr_min)
        return curr_min




    return cross_time_recur([], persons)

print("answer: ", cross_time([10, 20, 30]))
print("answer: ", cross_time([1, 2,5,8]))

#OK ALGO WORKS!!! BRUTE FORCE VERSION.


# algo with bitmasks:

'''

The approach is to use Dynamic programming. Before getting dive into dynamic programminc 
let’s see the following observation that will be required in solving the problem.

When any two people cross the bridge, then the fastest person crossing time will not be contributed 
in answer as both of them move with slowest person speed.

When some of the people will cross the river and reached the right side then only the 
fastest people(smallest integer) will come back to the left side.

Person can only be present either left side or right side of the bridge. Thus, if we maintain the 
left mask, then right mask can easily be calculated by setting the bits ‘1’ which is not present 
in the left mask. For instance, Right_mask = ((2n) – 1) XOR (left_mask).

Any person can easily be represented by bitmask(usually called as ‘mask’). When 
ith bit of ‘mask’ is set, that means that person is present at left side of 
the bridge otherwise it would be present at right side of bridge. 
For instance, let the mask of 6 people is 100101, which reprsents the person 1, 4, 6 are present 
at left side of bridge and the person 2, 3 and 5 are present at the right side of the bridge.

// C++ program to find minimum time required to
// send people on other side of bridge
#include <bits/stdc++.h>
using namespace std;
 
/* Global dp[2^20][2] array, in dp[i][j]--
   'i' denotes mask in which 'set bits' denotes
   total people standing at left side of bridge
   and 'j' denotes the turn that represent on 
   which side we have to send people either
   from left to right(0) or from right to 
   left(1)  */
int dp[1 << 20][2];
 
/* Utility function to find total time required
   to send people to other side of bridge */
int findMinTime(int leftmask, bool turn, int arr[], int& n)
{
 
    // If all people has been transfered
    if (!leftmask)
        return 0;
 
    int& res = dp[leftmask][turn];
 
    // If we already have solved this subproblem, 
    // return the answer.
    if (~res)
        return res;
 
    // Calculate mask of right side of people
    int rightmask = ((1 << n) - 1) ^ leftmask;
 
    /* if turn == 1 means currently people are at
     right side, thus we need to transfer
     people to the left side */
    if (turn == 1) {
        int minRow = INT_MAX, person;
        for (int i = 0; i < n; ++i) {
 
            // Select one people whose time is less
            // among all others present at right
            // side
            if (rightmask & (1 << i)) {
                if (minRow > arr[i]) {
                    person = i;
                    minRow = arr[i];
                }
            }
        }
 
        // Add that person to answer and recurse for next turn
        // after initializing that person at left side
        res = arr[person] + findMinTime(leftmask | (1 << person),
                                        turn ^ 1, arr, n);
    }
    else {
 
        // __builtin_popcount() is inbuilt gcc function
        // which will count total set bits in 'leftmask'
        if (__builtin_popcount(leftmask) == 1) {
            for (int i = 0; i < n; ++i) {
 
                // Since one person is present at left
                // side, thus return that person only
                if (leftmask & (1 << i)) {
                    res = arr[i];
                    break;
                }
            }
        }
        else {
 
            // try for every pair of people by
            // sending them to right side
 
            // Initialize the result with maximum value
            res = INT_MAX;
            for (int i = 0; i < n; ++i) {
 
                // If ith person is not present then
                // skip the rest loop
                if (!(leftmask & (1 << i)))
                    continue;
 
                for (int j = i + 1; j < n; ++j) {
                    if (leftmask & (1 << j)) {
 
                        // Find maximum integer(slowest
                        // person's time)
                        int val = max(arr[i], arr[j]);
 
                        // Recurse for other people after un-setting
                        // the ith and jth bit of left-mask
                        val += findMinTime(leftmask ^ (1 << i) ^ (1 << j),
                                                       turn ^ 1, arr, n);
                        // Find minimum answer among
                        // all chosen values
                        res = min(res, val);
                    }
                }
            }
        }
    }
    return res;
}
 
// Utility function to find minimum time
int findTime(int arr[], int n)
{
    // Find the mask of 'n' peoples
    int mask = (1 << n) - 1;
 
    // Initialize all entries in dp as -1
    memset(dp, -1, sizeof(dp));
 
    return findMinTime(mask, 0, arr, n);
}
 
// Driver program
int main()
{
    int arr[] = { 10, 20, 30 };
    int n = sizeof(arr)/sizeof(arr[0]);
    cout << findTime(arr, n);
    return 0;
}
'''

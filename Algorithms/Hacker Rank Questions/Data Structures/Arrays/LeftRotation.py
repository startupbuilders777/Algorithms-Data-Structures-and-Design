'''
Left Rotation
by saikiran9194
Problem
Submissions
Leaderboard
Discussions
Editorial 
A left rotation operation on an array of size  shifts each of the array's elements  unit to the left. 
For example, if  left rotations are performed on array , then the array would become .

Given an array of  integers and a number, , perform  left rotations on the array. Then print the 
updated array as a single line of space-separated integers.

Input Format

The first line contains two space-separated integers denoting the respective values of  (the number of integers) 
and  (the number of left rotations you must perform). 
The second line contains  space-separated integers describing the respective elements of the array's initial state.

Constraints

Output Format

Print a single line of  space-separated integers denoting the final state of the array after performing  left rotations.

Sample Input

5 4
1 2 3 4 5
Sample Output

5 1 2 3 4
Explanation

When we perform  left rotations, the array undergoes the following sequence of changes:

Thus, we print the array's final state as a single line of 
space-separated values, which is 5 1 2 3 4.

'''

# !/bin/python3

import sys


def leftRotation(a, d):
    # Complete this function
    newArr = [None] * len(a)
    for i in range(0, len(a)):
        newArr[i - d] = a[i]
    return newArr


if __name__ == "__main__":
    n, d = input().strip().split(' ')
    n, d = [int(n), int(d)]
    a = list(map(int, input().strip().split(' ')))
    result = leftRotation(a, d)
    print(" ".join(map(str, result)))

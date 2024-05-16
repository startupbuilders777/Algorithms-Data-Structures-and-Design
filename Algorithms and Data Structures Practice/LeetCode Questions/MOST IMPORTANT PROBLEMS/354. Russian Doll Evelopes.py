"""
354. Russian Doll Envelopes
Solved
Hard
Topics
Companies
You are given a 2D array of integers envelopes where envelopes[i] = [wi, hi] represents the width and the height of an envelope.

One envelope can fit into another if and only if both the width and height of one envelope are greater than the other envelope's width and height.

Return the maximum number of envelopes you can Russian doll (i.e., put one inside the other).

Note: You cannot rotate an envelope.

 

Example 1:

Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
Output: 3
Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).
Example 2:

Input: envelopes = [[1,1],[1,1],[1,1]]
Output: 1
 

Constraints:

1 <= envelopes.length <= 105
envelopes[i].length == 2
1 <= wi, hi <= 105
"""

class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        """

        Sort on one like Width or height.


        Then for the other one, do secondary sort on that one?

        Then 2 pointer?

        kaadanes??
        
        how about sort on "area" w*h then bigger area must contain smaller area..
        
        This is Longest increasing subsequence!
        After sorting on one dimension!!
        does it matter which dimension we sort on first?

        It failed because Envelopes of same size are interefering in the LIS algorithm 
        since LIS may pick envelopes of same width in its selection..

        to avoid this you can sort the other dimension in decreasing order apparently..
        
        The logic behind is, when you have envelopes which have the same width, 
        only one of them could be selected. Because you are going to solve this 
        problem by LIS, you have to sort their heights in DESC order to 
        guarantee their heights are NOT an Increasing Sequence, so that 
        at most only one of them could be selected in final LIS.
        """
        s_env = sorted(envelopes, key=lambda x: (x[0], -x[1]) ) 
        
        seq = list(map(lambda x: x[1], s_env))
        piles = []

        for i in range(len(seq)):
            placed = False
            # binary search this please..
            
            x = bisect.bisect_left(piles, seq[i])

            if x < len(piles) and seq[i] <= piles[x]:
                piles[x] = seq[i]
            else:
                # start a new pile that contains a big number!
                piles.append(seq[i])
        return len(piles)


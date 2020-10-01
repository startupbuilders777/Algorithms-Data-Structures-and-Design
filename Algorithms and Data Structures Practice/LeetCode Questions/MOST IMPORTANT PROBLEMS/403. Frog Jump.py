'''
403. Frog Jump
Hard

1200

115

Add to List

Share
A frog is crossing a river. The river is divided 
into x units and at each unit there may or may not exist 
a stone. The frog can jump on a stone, but it must not jump into the water.

Given a list of stones' positions (in units) in sorted 
ascending order, determine if the frog is able to cross the 
river by landing on the last stone. Initially, the frog is on
the first stone and assume the first jump must be 1 unit.

If the frog's last jump was k units, then its next jump must be 
either k - 1, k, or k + 1 units. Note that the frog can only jump 
in the forward direction.

Note:

The number of stones is ≥ 2 and is < 1,100.
Each stone's position will be a non-negative integer < 231.
The first stone's position is always 0.
Example 1:

[0,1,3,5,6,8,12,17]

There are a total of 8 stones.
The first stone at the 0th unit, second stone at the 1st unit,
third stone at the 3rd unit, and so on...
The last stone at the 17th unit.

Return true. The frog can jump to the last stone by jumping 
1 unit to the 2nd stone, then 2 units to the 3rd stone, then 
2 units to the 4th stone, then 3 units to the 6th stone, 
4 units to the 7th stone, and 5 units to the 8th stone.
Example 2:

[0,1,2,3,4,8,9,11]

Return false. There is no way to jump to the last stone as 
the gap between the 5th and 6th stone is too large.


'''


class Solution:
    '''
    In this case, 
    Recursive DP is the same as traversing a graph with DFS using a visited set.
    Please memorize this relationship.
    
    Visited set -> is the DP array for dfs, which allows us to optimize to linear time from factorial time.
    
    When you do a DP question -> think about GRAPH traversal as a possible means to a solution as well
    because its the same thing essentially. 
    '''
    def canCross(self, stones):
        # just do a dfs on this graph
        # with a visted set that is for checking if we saw the state before. 

        p_to_idx = {}
        
        for idx, stone in enumerate(stones):
            p_to_idx[stone] = idx
            
        # just need visited set. 
        visited = set()
        
        def recursive(i, k):
            nonlocal visited
            if i == len(stones)-1:
                return True
            
            curr_pos = stones[i]
            visited.add((i, k))
            res = False
            
            for incr in [-1, 0, 1]:
                if curr_pos + k + incr in p_to_idx:
                    nxt_node = p_to_idx[curr_pos + k + incr]
                    
                    
                    
                    if nxt_node > i and (nxt_node,  k + incr) not in visited:
                        res = recursive(nxt_node, k+incr)                    
                        if res: # EARLY RETURN FOR PERFORMANCE BOOST
                            return True
            return res
        return recursive(0, 0)

    
    # FORWARD DP SOLUTION.
    # IT TLE'D due to its BFS type search
    def canCrossForwardDP(self, stones: List[int]) -> bool:
        N = len(stones)
        # from position 0, you are only allowed to jump with size 1!
        # then you can append all the possible jumps you can do from that timestep. 
        # and keep appending. 
        '''
        We need a reverse lookup map for question
        '''
        p_to_idx = {}
        
        for idx, stone in enumerate(stones):
            p_to_idx[stone] = idx 
        
        if N == 0:
            return False
        if N == 1:
            return True
        
        allowed_jumps = [[] for i in range(N)]
        # print("p to idx", p_to_idx)
        
        if stones[1] == stones[0] + 1:
            allowed_jumps[1].append(1)
        
        for i  in range(1, N):
            curr_pos = stones[i]
            jumps = allowed_jumps[i]
            # print("allowed jumps", jumps)
            
            for j in jumps:    
                for incr in [-1, 0, 1]:
                    if curr_pos + j + incr in p_to_idx:
                        idx = p_to_idx[curr_pos + j + incr]

                        if idx == N-1:
                            return True
                        elif idx > i:
                            allowed_jumps[idx].append(j + incr)
                            
        if len(allowed_jumps[N-1]) > 0:
            return True
        else:
            return False
        

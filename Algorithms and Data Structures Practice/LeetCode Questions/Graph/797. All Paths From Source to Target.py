'''
Given a directed, acyclic graph of N nodes.  Find all possible paths from node 0 to node N-1, and return them in any order.

The graph is given as follows:  the nodes are 0, 1, ..., graph.length - 1.  graph[i] is a list of all nodes j for which the edge (i, j) exists.

Example:
Input: [[1,2], [3], [3], []] 
Output: [[0,1,3],[0,2,3]] 
Explanation: The graph looks like this:
0--->1
|    |
v    v
2--->3
There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.
Note:

The number of nodes in the graph will be in the range [2, 15].
You can print different paths in any order, but you should keep the order of nodes inside one path.
'''

# DYNAMIC PROGRAMMING SOLUTION TOP-DOWN!! BY USING @lru_cache(maxsize=None)
# THIS SOLUTION IS BAD BECAUSE WE ARE NOT USING DEQUE AND APPENDLEFT, 
# LIST MERGING AND INSERTION TO FRONT IS O(N)!!!

#The two approach might have the same asymptotic time 
#complexity. However, in practice the DP approach is 
#slower than the backtracking approach, since we copy the intermediate paths over and over.
#Note that, the performance would be degraded further, 
#if we did not adopt the memoization technique here.
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        
        # apply the memoization
        @lru_cache(maxsize=None)
        def dfs(node):
            
            if node == len(graph) - 1:
                return [[len(graph) - 1]]
            
            kids  = graph[node]
            
            # all paths from node to target. 
            paths = []
            
            for kid in graph[node]:
                res = dfs(kid)
                # add node to front of each result!
                for result in res:
                    paths.append([node] + result)
            
            return paths
        return dfs(0)
        

class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:

        target = len(graph) - 1
        results = []

        def backtrack(currNode, path):
            # if we reach the target, no need to explore further.
            if currNode == target:
                results.append(list(path))
                return
            # explore the neighbor nodes one after another.
            for nextNode in graph[currNode]:
                path.append(nextNode)
                backtrack(nextNode, path)
                path.pop()
        # kick of the backtracking, starting from the source node (0).
        path = deque([0])
        backtrack(0, path)

        return results
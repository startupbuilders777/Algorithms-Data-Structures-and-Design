'''

207. Course Schedule

There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to 
first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible 
for you to finish all courses?

Example 1:

Input: 2, [[1,0]] 
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.
Example 2:

Input: 2, [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
Note:

The input prerequisites is a graph represented by a list of edges, not adjacency 
matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.
'''

from collections import defaultdict, deque

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        # You cannot use bipartite finding to find 
        # cycles when there is no bipratite.
        # No bipartite only means there exists an odd cycle
        # Nothing about even cycles. 
        
        # How about inorder outorder topo-sort with bfs?
        # find nodes with inorder and out-order
        
        
        # Keep processing nodes with in-order 0 right!
        # when we process children. find children 
        # that have inorder 0 once parent is removed.
        
        g = defaultdict(set)
        inorder_count = defaultdict(int)    
        # Init
        for c in range(numCourses):
            inorder_count[c] = 0
            
        for req in prerequisites:
            g[req[1]].add(req[0])
            inorder_count[req[0]] += 1
        
        print("inorder count")
        
        root_nodes = [k for (k,v) in  inorder_count.items() if v == 0]
        print("root nodes", root_nodes)
        
        print("G", g)
        print("Inorder count", inorder_count)
        
        d = deque(root_nodes)
        visited = set()
        while d:
            node = d.popleft()
            
            visited.add(node)
            
            children = g[node]
            for c in children:
                inorder_count[c] -= 1
                if(inorder_count[c] == 0):
                    d.append(c)
        
        # If you cant visit all nodes from root nodes, then there is a cycle 
        # in directed graph. 
        
        return len(visited) == numCourses

                   
    
    def canFinishWithDFSColors(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        
        g = defaultdict(set)
            
        for req in prerequisites:
            g[req[1]].add(req[0])
        
        def has_cycle_directed(node, g, colors):
            # seen.add(node)
            colors[node] = "G" # Grey being processed
            
            for c in g[node]:
                if colors.get(c) is None and has_cycle_directed(c, g, colors):
                      return True
                elif colors.get(c) == "G":
                    # We are processing this node but we looped back around somehow
                    # so cycle
                    return True
                else: 
                    # The node we are processing has already been processed. 
                    continue 
                    
            colors[node] = "B" # Black
            return False
        
        colors = {}
        # Initiall all nodes are White, which is None in this case.
        # If it is being processed, Grey
        # If its processed -> Black
        
        for i in range(numCourses):
            if(colors.get(i) is None and has_cycle_directed(i, g, colors)):
                return False     
        return True

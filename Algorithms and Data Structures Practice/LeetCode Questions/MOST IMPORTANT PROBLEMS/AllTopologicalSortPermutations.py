

'''

Given a directed graph.

Create all topological sorts, and not just 1!


Also the graph can be disconnected forest. 
So allow permutations here!


 
'''

# Here is a directed graph. 
courses = [[1, 2], [2, 3], [4, 3  ], [3, 6], [5, 7]]

from collections import defaultdict


def getTopos(courses, N):
    # there is range(N) courses
    
    g = defaultdict(list)
    
    all_courses = range(N)    
    for prev, next in courses:
        g[prev].append(next)
    
    
    print("GRAPH IS ", g)
    
    # Do three color processing. 
    '''
    Find all nodes with 
    indegree == 0 -> then remove it from graph.  
    see nodes which have indegree 0 after that and repeat.
    
    to remove in O(1) -> use map of sets, and keep track of reverse graph. 
    '''
    
    colors = {}
    
    # W -> not processed, G -> processing, B -> processed 
    for n in range(N):
        colors[n] = "W"
    
    
    def dfs(node):
        
        colors[node] = "G"    
        
        # We need to know what are all our possible set of choices at each step,
        # then place it accordingly. so we need to keep a list of all possible choices. 
        kids = g[node]
        
        topos = []
        for k in kids:
            # get all topos per kid. 
            # but it depends on how we place them rite? 
            # We want all topos but excluding our parent node rite!!!
            # the kids are just members that can be used for topo now that is it. 
            # we still have a large candidate pool to TOPO!
            
            
            dfs(k)
            
        
        
        

        
        
        
    
    

getTopos(courses, 9)
     
    
    
    
    



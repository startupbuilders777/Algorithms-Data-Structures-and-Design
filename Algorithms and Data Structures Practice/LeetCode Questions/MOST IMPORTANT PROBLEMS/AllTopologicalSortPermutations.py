

'''

Given a directed graph.

Create all topological sorts, and not just 1!


Also the graph can be disconnected forest. 
So allow permutations here!

'''

# Here is a directed graph. 
courses = [[1, 2], [2, 3], [4, 3  ], [3, 6], [5, 7]]

coursesB = [[5, 0], [4,0], [4,2], [5, 1], [2,3], [3,1] ] 

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
    
    # Can memoize paths returned!
    
    def dfs(node, s):
        
        # colors[node] = "G"    
        s.add(node)
        
        # We need to know what are all our possible set of choices at each step,
        # then place it accordingly. so we need to keep a list of all possible choices. 
        kids = g[node]        
        numKids = len(kids)
        
        all_paths = []
        
        for offset in range(numKids):
            copy_set = s.copy()
            for k in range(numKids):
                # get all topos per kid. 
                # but it depends on how we place them rite? 
                # We want all topos but excluding our parent node rite!!!
                # the kids are just members that can be used for topo now that is it. 
                # we still have a large candidate pool to TOPO!
                
                kid = kids[(k + offset) % numKids]
                if kid not in copy_set:
                    paths = dfs(kid, copy_set)
                    all_paths += paths
                    
                # after we process a kid, do we set back to w?
            
        # you need to add all the points you  saw to copy set. 

        if all_paths == []:
            all_paths = [[node]]
        else:  
            for path in all_paths:
                path.append(node)
        
        a_path = all_paths[0]
        for node in a_path:
            s.add(node)        
        
        return all_paths
    
    
    total_perms = []    
    for offset in range(N):
        s = set()
        all_paths = []
        
        for n in range(N):
            node = (n + offset) % N
            if node not in s:
                paths = dfs(node, s)
                
                print('ALL THE PATHS WE SAW WITH STARTING node', node, paths)
                
                # all paths will have visited the same nodes.
                # now we get the paths from another tree in the forest. 
                # and we can permutate those? -> or just add to front of all paths we have!
                # so cross multiplication. 
                
                a_path = paths[0]
                for point in a_path:
                    s.add(point)
                
                if(all_paths == []):
                    all_paths = paths 
                else: 
                    # cross it.
                    new_paths = []
                    for start in all_paths: 
                        for end in paths:
                            new_paths.append(start + end)
                            
                    print("crossed paths", new_paths) 
                    all_paths = new_paths
        
        total_perms += all_paths
        print("--------------------------- total perms", total_perms)
        
        
    
    return list(map(lambda x: x[::-1], total_perms))

            
    # return result[::-1]

'''

Answer should be 

5 4 2 3 1 0 
or
5 4 0 2 3 1


but we need all permutations rite!

so we need the independent paths, but not really.
i think you cant UNLESS, you change forloop, 
add another forloop -> loop thru a diff kid each time. 



but also:

you could ->

find all 0 nodes 

add that remove recurse. 

then try a diff 0 node, find that recurse
on 1 nodes, 2-nodes, 3-nodes 

but we add it to set so that we dont have duplicating paths? 


Could we also...
figure out what nodes become available once we add. 




'''
print("THE ANSWER IS", getTopos(coursesB, 6))

     
    
    
    
    



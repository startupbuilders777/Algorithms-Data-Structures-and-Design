'''
[10 marks] Given a weighed DAG G (a directed acyclic graph with weights on the
edges), and the weight of the longest path in G. That is, Find the maximum, over all
pairs of vertices u and v, of the maximum-weight path from u to v.
The weight of the path here is dened as the total of the edge weights on the path.
You may suppose that the edge weights are non-negative. Your algorithm only needs
to output the maximum path weight, but not the actual path.
'''


'''



Process nodes 
Do a topological sort of the nodes in the graph.
Process the nodes individually from right to left 
(The nodes on the right are topologically after the nodes on the left)
Processing the nodes in this way allows for bottom up dynamic programming,
because the subproblems will be the longest weight paths from the specified node.
Nodes at the beginning of the topological sort will reuse these subproblems as 
they attempt to travel to every neighbour (so if that neighbour has been processed)
we can stop travelling and use that neighbour's computed longest path value. 

We will be taking the max of all the weights for the paths that come out of a node and 
save that as a subproblem in our dynamic programming and process from right to left, and 
seperately also be saving the largest weight we have seen so far in our processing of the 
nodes. We return this largest weight we have seen so far when we have processed the final node (which will be 
the beginning node in our dynamic programming).



For the dynamic programming we have the following recurrence (on the topological ordering of the vertices stored in an array): 
Let V be the vertices and E be the edges, and |V| be the number of vertices, and |E| be the number of edges

Let N be a vertex, and i be the index of N in the topological ordering array. 
Let M be a vertex, and j be the index of M in the topological ordering array. 
Let w be the function to get the weight of an edge (i, j), where i and j are vertices

        / 0 if i = |V|
T(i) =  | for all out neighbours M who have index j, max(T(j)) + w(N, M) 


Pseudocode:

//g is a graph

maxweightpathindag(g):

    arr = topologicalsort(g) //get array with topological sorted nodes
    
    nodetoindexintopologicalsort = [] //this will be same as arr but for reverse lookup

    for index, value in arr:
         nodetoindexintopologicalsort[value] = index
    


    maxweightsofar = 0

    T = Array with |V| elements initialized to 0

    for i = arr.length to 0:
        if(i == arr.length):
            T[i] = 0
        else:
            //Get max over all out neighbours
            node = arr[i]

            if(g[node].outchildren().length == 0) 
                T[i] = 0
                continue
            
            maxPath = 0
            maxChild = undefined
            for child in g[node].outchildren(): 
                indexofchild = nodetoindexintopologicalsort[child]
                childpathweight = T[indexofchild]
                if(childpathweight > maxPath)
                    maxChild = child 
                    maxPath = childpathweight


            T[i] = maxPath + weightofedge((node, maxChild))
            maxweightsofar = max(T[i], maxweightsofar)

            
    return maxweightsofar


Runtime:

Topological Sort is O(V+E)




            






'''





def maxWeightPathInDAG(g):

    def topologicalSort(g):
        
    



















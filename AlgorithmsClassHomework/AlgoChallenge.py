'''
Challenge #10: Paths in graphs

Consider the following problem: given an unweighted directed graph G, 
and nodes s and t, decide whether there exist at least two directed 
SIMPLE paths from s to t of different lengths. Is the problem 
solvable in polynomial time, or is it NP-complete?


I offer $100 and an automatic 100 for the course 
if you come up with an original solution.

BFS in a graph. 
Finds the shortest distance to each node.
Call the distance to get to T, X. 

Some X-3 Length
Some X-2 Length
Some X-1 Length
One node is X length

Save some nodes. 
Save all nodes that are (X-1) in length


Now do a BFS, from every node. 


OKAY NEW ALGORITHM.

COUNT ALL THE SHORTEST PATHS BETWEEN 2 NODES
This was done in A7.
We can think of the shortest path between 2 nodes as a flow.
The flow has segments which are each shortest paths to get from s to t.
We count all the paths in a segment from s to t. 
We take one of those paths and try to augment its length.
then use that to build a path that is sligthly longer.





BFS(G, s)
    for all u in V do
        dist[u] := infinity;
        numpaths[u] := 0;
        pi[u] := nil;
    
    dist[s] := 0;
    numpaths[s] := 1;
    
    Q := empty queue;
    
    enqueue(Q, s);
    while (Q is nonempty) do
        u := dequeue(Q);
        
        for all edges (u,v) in E do
            if dist[v] = infinity then
                enqueue(Q,v)
                dist[v] := dist[u]+1;
                pi[v] := u;
                numpaths[v] := numpaths[u];
            
            else if dist[v] = dist[u]+1 then
                numpaths[v] := numpaths[u]+numpaths[v];



The running time is still O(jV j + jEj) because we have only added a constant number
of operations in the loops already analyzed for BFS.

'''


'''

The following is the algorithm :


Run bfs on the graph. Find the shortest path between point s and t.
Remove the edges (not the vertices) in between s and t that were part of the shortest path, and put these edges in set 1.
Run bfs between s and t again. Check if there is a another shortest path of same length as the one we discovered. If there is, 
remove it, and put these points in set 2. 
Keep doing this, and removing intermediate edges that form shortest paths of equal length.

All the shortest paths from s to t are removed. Let X be the union of these simple paths ( and let the graph be directed from s to t).
X is a DAG from s to t.

The edges in G-X are a bunch of small subgraphs. They are a forest of edges. 

The core idea is: We need to add a path from G-X that connects to 2 points in the DAG, point x and point y. 
If point x and point y can be found, the longer path (or equal length path is) is: [s->x, x->y, y->t]

This path is longer than the st paths in the DAG because, otherwise, it would have been added by the series of BFS's at the beginning of the algorithm.
(If you dont do the BFS's at the beginning, do no add the path.)



def two_paths(G, s, t):
    
    DAGVertices = set()
    
    for all u in V do
        dist[u] := infinity;
        pi[u] := nil;
    
    dist[s] := 0;
    
    Q := empty queue;
    
    enqueue(Q, s);

    Do BFS, add st path vertices in DAG vertices.
    Now we have vertices in DAGVertices.

    shortestPathLength = length(st-path);

    verticesToTest = set() = G vertices - DAGVertices
    filter out vertices with degree 1 or 0 from verticesToTest

    testedVertices = set()
    

    //START 2 STEP BFS

    for vertex in verticesToTest: 
        //do bfs with vertex and find 2 points, X and Y, that intersect with DAG vertices.

        X = None 
        Y = None


        for all u in V do
            dist[u] := infinity;
            pi[u] := nil;
        
        dist[s] := 0;
        
        visited = set()

        Q := empty queue;
        
        enqueue(Q, s);

        while (Q is nonempty) do
            u := dequeue(Q);

            if(u in visited):
                continue
            else: 
                visited.add(u)


            if(u in DAGVertices):
                if(X is None):
                    X = u;
                    continue
                if(Y is None):
                    Y = u
                    break;

            for all edges (u,v) in E do
                enqueue(Q,v)
                dist[v] := dist[u]+1;
                pi[v] := u;

        //DONE BFS FOR A VERTEX. CHECK!
        // PATH IS [S->X, X->COMMONPARENT, COMMONPARENT->Y, Y->T]

        COMMON PARENT = find common parent of X and Y (use parents array and go backward)
        NEWPATH = <create path and check its length. The path is [S->X, X->COMMONPARENT, COMMONPARENT->Y, Y->T]>

        if NEWPATH length is same as shortestPathLength:
            ADD VERTICES OF NEWPATH to DAGVertices
            verticesToTest = verticesToTest - visited  //This works because vertices in a connected component dont need to be checked. 
                                                        //TODO: CHECK THIS THOROUGHLY.

        else:
            RETURN TRUE (2 PATHS ARE NEWPATH AND STPATH)
        





        










    

    
    
    
    








BFS(G, s)
    for all u in V do
        dist[u] := infinity;
        numpaths[u] := 0;
        pi[u] := nil;
    
    dist[s] := 0;
    numpaths[s] := 1;
    
    Q := empty queue;
    
    enqueue(Q, s);
    while (Q is nonempty) do
        u := dequeue(Q);
        
        for all edges (u,v) in E do
            if dist[v] = infinity then
                enqueue(Q,v)
                dist[v] := dist[u]+1;
                pi[v] := u;
                numpaths[v] := numpaths[u];
            
            else if dist[v] = dist[u]+1 then
                numpaths[v] := numpaths[u]+numpaths[v];







'''


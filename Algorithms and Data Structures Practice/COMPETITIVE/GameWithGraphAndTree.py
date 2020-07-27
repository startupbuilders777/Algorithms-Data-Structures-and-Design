'''
TopCoder problem "GameWithGraphAndTree" used in Member Single Round Match 474 (Division I Level Three)

Problem Statement
    	
Little Dazdraperma likes graphs a lot. Recently she invented a new game for one person with graphs. 
Given a connected undirected graph with N vertices and a tree with N nodes (see the notes for 
definitions of all unclear terms from graph theory), she tries to place that tree on the graph in the following way:

Each node of the tree is put into correspondence with a vertex of the graph. Each node 
then corresponds to one vertex and each vertex corresponds to one node.
If there is an edge between two nodes of the tree then there must be an edge between the 
corresponding vertices in the graph.
Now Dazdraperma wonders how many ways are there to do such placement. Two placements are 
considered equal if each node of the tree corresponds to the same vertex of the graph in
both placements. Return this number modulo 1000000007.



The graph will be represented as String[] graph where j-th character in i-th element
will be 'Y' if there is an edge between vertices i and j and 'N' otherwise. 
The tree will be given in the same way in String[] tree.
 
Definition
    	
Class:	GameWithGraphAndTree
Method:	calc
Parameters:	String[], String[]
Returns:	int
Method signature:	int calc(String[] graph, String[] tree)
(be sure your method is public)
    
 
Notes
-	For the purpose of this problem, an undirected graph can be treated as a set of 
    vertices and a set of edges, where each edge establishes a bidirectional connection 
    between two different vertices.

-	A path between two different vertices A and B in a graph G is a 
    sequence of 2 or more vertices v[0] = A, v[1], ..., v[L-1] = B, such that 
    there's an edge in G between each two adjacent vertices in this sequence.
    
-	A graph G is connected if there's a path between each two different vertices of G.
-	A graph G is a tree if it is connected and contains exactly V-1 edges, 
    where V is the total number of vertices in G.
 
Constraints
-	graph will contain between 1 and 14 elements, inclusive.
-	tree will contain the same number of elements as graph.
-	Each element of graph will contain the same number of characters as the number of elements in graph.
-	Each element of tree will contain the same number of characters as the number of elements in tree.
-	Each character in each element of graph will be either 'Y' or 'N'.
-	Each character in each element of tree will be either 'Y' or 'N'.
-	For each i, i-th character of i-th element in both graph and tree will be equal to 'N'.
-	For each i and j, j-th character of i-th element in graph will be equal to i-th character of j-th element.
-	For each i and j, j-th character of i-th element in tree will be equal to i-th character of j-th element.
-	graph will represent a connected graph.
-	tree will represent a tree.
 
Examples
0)	
    	
{"NYN",
 "YNY",
 "NYN"}
{"NYY",
 "YNN",
 "YNN"}
Returns: 2
Vertex 1 of the graph must correspond to node 0 of the tree. 
There remain 2 possible ways to map vertices 0 and 2 of the graph.
1)	
    	
{"NYNNN",
 "YNYYY",
 "NYNYY",
 "NYYNY",
 "NYYYN"}
{"NYNNN", 
 "YNYNN",
 "NYNYN",
 "NNYNY",
 "NNNYN"}
 
Returns: 12
In this case vertex 0 of the graph can correspond only to nodes 0 and 4 of the tree. 
If it corresponds to 0, vertex 1 of the graph must correspond to node 1 of the tree. 
All other vertices can be mapped in any way, so there are 3! possible mappings. 
There are also 3! mappings when vertex 0 of the graph corresponds to node 4 
of the tree. The total number of mappings is 2*3!=12.
2)	
    	
{"NYNNNY",
 "YNYNNN",
 "NYNYNN",
 "NNYNYN", 
 "NNNYNY",
 "YNNNYN"}
{"NYNNYN",
 "YNNYNY",
 "NNNNYN",
 "NYNNNN",
 "YNYNNN",
 "NYNNNN"}
Returns: 0
There are no possible mappings in this test case.
3)	
    	
{"NYNNYN",
 "YNNYNY",
 "NNNNYN",
 "NYNNNN",
 "YNYNNN",
 "NYNNNN"}
{"NNNYYN", 
 "NNYNNN",
 "NYNNYY", 
 "YNNNNN",
 "YNYNNN",
 "NNYNNN"}
Returns: 2
The graph can also be a tree.
4)	
    	
{"NYNNNYNNY",
 "YNNNNNNYN",
 "NNNNYYNYY",
 "NNNNNYNNY",
 "NNYNNNYNY",
 "YNYYNNNYN",
 "NNNNYNNYN",
 "NYYNNYYNN",
 "YNYYYNNNN"}
{"NNYNNNYYN",
 "NNNNYNNNN",
 "YNNNNNNNN",
 "NNNNNNYNN",
 "NYNNNNNYY",
 "NNNNNNNNY",
 "YNNYNNNNN",
 "YNNNYNNNN",
 "NNNNYYNNN"}
Returns: 90

'''


'''
GameWithGraphAndTree

This problem is solved by very tricky DP on the tree and subsets. 
We are required to find number of mappings of the tree on the graph. 
First of all, we choose root of the tree because it is easier to handle 
rooted tree. Clearly, we should consider all possible submapping of all 
subtrees on all vertex-subsets of the graph. The number of these submapping 
is huge, so we have to determine which properties of these submappings are 
important for extending the mapping. It turns out that these properties are: 

1. Subtree denoted by its root vertex v. Necessary to check the outgoing edge mapping later. 
2. Vertex of graph p which is the image of v. Again: necessary to check the mapping of added edge. 
3. The full image of v-subtree in graph — set s of already mapped vertices in graph. 
    Necessary to maintain bijectivity of mapping. Therefore we define state domain (v,p,s)->GR. 
    
GR is number of submappings with the properties we are interested in. 

To combine results of sons in tree we need to run another "internal" DP. 
Remember that internal DP is local for each vertex v of the tree. 
The first parameter will be number of sons already merged — this is quite standard. 
Also we'll use additional parameters p and s inside. The state domain is (k,p,s)->IR 
where IR is the number of submappings of partial v-subtree on graph with properties: 

1. The vertex v and subtrees corresponding to its first k sons are being mapped (called domain). 
2. Image of v is vertex p in graph. 
3. The full image of mapping considered is s — subset of already used vertices. 
    The transition of this internal DP is defined by adding one subtree corresponding to 
    k-th son to the domain of mapping. For example, if w is the k-th son, then 
    we add global state GR[w,q,t] to internal state IR[k,p,s] and get internal state 
    IR[k+1,p,s+t]. Here we must check that there is an edge p-q in the graph and that 
    sets s and t have no common elements. The combinations considered in GR[w,q,t] and IR[k,p,s] 
    are independent, so we can just add their product to the destination state. The answer of 
    internal DP is IR[sk,p,s] which is stored as a result GR[k,p,s] of global DP. 
    This is correct solution of this problem. Unfortunately, it runs in O(4^N * N^3) 
    if implemented like it is in the code below. Of course it gets TL. You need to 
    optimize the solution even further to achieve the required performance. 
    The recipe "Optimizing DP solution" describes how to get this solution accepted.

int gres[MAXN][MAXN][SIZE];                                //global DP on subtrees: (v,p,s)->GR
int ires[MAXN][MAXN][SIZE];                                //internal DP: (k,p,s)->IR
 
void DFS(int v) {                                          //solve DP for v subtree
  vis[v] = true;
  vector<int> sons;
  for (int u = 0; u<n; u++) if (tree[v][u] && !vis[u]) {   //visit all sons in tree
    DFS(u);                                                //calculate gres[u,...] recursively
    sons.push_back(u);                                     //ans save list of sons
  }
  int sk = sons.size();
  memset(ires[0], 0, sizeof(ires[0]));                     //base of internal DP
  for (int p = 0; p<n; p++) ires[0][p][1<<p] = 1;          //one-vertex mappings v -> p
  for (int k = 0; k<sk; k++) {                             //iterate through k &mdash; number of sons considered
    int w = sons[k];
    memset(ires[k+1], 0, sizeof(ires[k+1]));               //remember to clear next layer
    for (int p = 0; p<n; p++) {                            //iterate through p &mdash; image of v
      for (int s = 0; s<(1<<n); s++)                       //iterate through s &mdash; full image of current domain
        for (int q = 0; q<n; q++) if (graph[p][q])         //consider adding mapping which maps:
          for (int t = 0; t<(1<<n); t++) {                 //w -> q;  w-subtree -> t subset;
            if (s & t) continue;                           //do not break bijectivity
            add(ires[k+1][p][s^t], mult(ires[k][p][s], gres[w][q][t]));
          }                                                //add product of numbers to solution
    }
  }
  memcpy(gres[v], ires[sk], sizeof(ires[sk]));             //since partial v-subtree with k=sk is full v-subtree
}                                                          //we have GR[v,p,s] = IR[sk,p,s]
...
    DFS(0);                                                //launch DFS from root = 0-th vertex
    int answer = 0;                                        //consider all variants for i &mdash; image of 0
    for (int i = 0; i<n; i++) add(answer, gres[0][i][(1<<n)-1]);
    return answer;                                         //sum this variants up and return as an answer
  }
};
'''



'''
The solution of this problem was discussed in detail in recipe 
"Commonly used DP state domains". Let size(v) be size of v-subtree. 
The following constraints hold for any useful transition: 1,2) 
size(p) = |s|; p in s; 3,4) size(q) = |t|; q in t; 5) t and s have no 
common elements; All the other iterations inside loops are useless because 
either ires[k][p][s] or gres[son][q][t] is zero so there is not impact of addition.

The 5-th constraint gives a way to reduce O(4^N) to O(3^N) by applying the 
technique described in recipe "Iterating Over All Subsets of a Set". 
Since t and s do not intersect, t is subset of complement to s and loop 
over t can be taken from that recipe. The time complexity is reduced to O(3^N * N^3).

The easiest way to exploit constraints 1 and 2 is to check ires[k][p][s] to 
be positive immediately inside loops over s. The bad cases for which constraints 
are not satisfied are pruned and the lengthy calculations inside do not happen 
for impossible states. From this point is it hard to give precise time complexity. 
We see that number of possible sets s is equal to C(n, size(p)) where C is binomial 
coefficient, and this binomial coefficient is equal to O(2^n / sqrt(n)) in worst case. 

So the time complexity is not worse than O(3^N * N^2.5).

The other optimizations are not important. The cases when q belongs to set s are pruned. Also there is a check that gres[son][q][t] is positive. The check is faster than modulo multiplication inside loop over t so let it be. The loop over t remains the most time-consuming place in code. Ideally this loop should iterate only over subsets with satisfied constraint 3 — it should accelerate DP a lot but requires a lot of work, so it's better to omit it. Here is the optimized piece of code:

    for (int p = 0; p<n; p++) {
      for (int s = 0; s<(1<<n); s++) if (ires[k][p][s])                   //check that result is not zero &mdash; prune impossible states
        for (int q = 0; q<n; q++) if (graph[p][q]) {
          if (s & (1<<q)) continue;                                       //prune the case when q belongs to set s
          int oth = ((1<<n)-1) ^ s;                                       //get complement to set s
          for (int t = oth; t>0; t = (t-1)&oth) if (gres[son][q][t])      //iterate over non-empty subsets of oth
            add(ires[k+1][p][s^t], mult(ires[k][p][s], gres[son][q][t])); //do calculation only for possible gres states
        }
    }
'''
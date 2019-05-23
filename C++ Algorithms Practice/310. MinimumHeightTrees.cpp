/*
310. Minimum Height Trees
Medium

938

59

Favorite

Share
For an undirected graph with tree characteristics, we can choose any node as the root.
 The result graph is then a rooted tree. Among all possible rooted trees, those with minimum 
 height are called minimum height trees (MHTs). Given such a graph, write a function to find all the MHTs and return a list of their root labels.

Format
The graph contains n nodes which are labeled from 0 to n - 1. You will be given the number n and a list of undirected edges (each edge is a pair of labels).

You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

Example 1 :

Input: n = 4, edges = [[1, 0], [1, 2], [1, 3]]

        0
        |
        1
       / \
      2   3 

Output: [1]
Example 2 :

Input: n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]

     0  1  2
      \ | /
        3
        |
        4
        |
        5 

Output: [3, 4]


*/

//MY ACCEPTED SOLUTION: 

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue> 

class Solution {
public:
    
    int getHeight(int currNode, 
                  unordered_map<int, 
                  unordered_set<int> > & tree, 
                  unordered_set<int> & visited) {
        
       
            
        int height = 0; //leafs have this height
        try {
        for(auto child : tree.at(currNode)) {
            // cout << "Parent " << currNode << " has child " << child << endl;
            if(visited.find(child) == visited.end()) { 
                visited.insert(child);
                
                int theHeight = 1 + this->getHeight(child, tree, visited);
                
                if(theHeight > height){
                    height = theHeight;
                }
                
            }
        } 
        } catch( ... ) {
            return 0;
        }
        
        
         // cout << "GET HEIGHT FOR " << currNode << " whic is "  << height << endl;
  
        
        return height;
    }
    
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        
        //Create a map that goes from start => end 
        // in other words adjcency list representation?
        // yee do it to practice data structures.
        
        //Choose any node as the root. Then graph becomes rooted tree
        //Find all minimum height trees.
        
        //DFS to get height of tree. Sure. 
        // Do N dfs's and get height for all treess like that. 
        
        
        // Or we can BFS from one node to all other nodes. 
        // This node is the "leaf" node of the tree. 
        // return nodes that are 50% away. rite? nah doesnt work.
        
        // Fastest solution is to iteratively remove a leaf from the outside, and go in. 
        // a leaf is on the outside if it has only one connection to the graph. 
        // keep doing that until you have 
        // A) only 1 node left, or
        // B) removing the node at that level causes the tree to disappear (aka one connection)
        // C) Cant remove nodes because no nodes only have 1 connection
        
        
        //create map that maps int to set
        
        //map<int, set<int> > m;
        unordered_map< int, unordered_set<int> > m;
        
        
        if(edges.size() == 0) {
            return vector<int>{0};
        }
        
        // This way to put edges in map was incorrect because didnt 
        // realize each vector element is 2 size array
        /*
        for(auto it=edges.begin(); it != edges.end(); ++it) {
            int startNode = it - edges.begin();
            set<int> neighbors;
            for(auto jt=it->begin();  jt!= it->end(); ++jt){
                
                neighbors.insert(*jt);
            }
         
            //Hard way to insert into map is the following: 
            m.insert(pair<int, set<int> >(startNode, neighbors));
        }
        */
        // ALSO USE UNORDERED_MAP. it has amortized lookup of O(1) VS O(lg n) for regular maps.
        
        for(auto &it : edges){
            auto first = it[0];
            auto second = it[1];
            m[first].insert(second);
            m[second].insert(first);
        }
        
        
        for(auto mit=m.begin(); mit != m.end(); ++mit){
            
            /// cout << "FOR KEY: " << mit->first<< endl;
            for(auto setElement : mit->second){
             //   cout << "set element is " << setElement << " ";  
            }
            cout << endl;
        } 
        
         /*     
         cout << "map initially looks like this: " << endl;
         for(auto kv: m){
                        cout << "key " << kv.first;
                        for(auto v: kv.second) {
                            cout << " has val " << v << " " ;
                        }
                        cout << endl;
                    }
        
        */
        
        // Okay time to do DFS and get min of all trees. 
        
        /*
        This is how you get keys and vals from map btw:
        
        std::vector<Key> keys;
        keys.reserve(map.size());
        std::vector<Val> vals;
        vals.reserve(map.size());

        for(auto kv : map) {
            keys.push_back(kv.first);
            vals.push_back(kv.second);  
        } 
        */
        
        //THIS SOLUTION IS TOO SLOW BECAUSE IT DOES DFS ON EVERY ROOT WHICH IS BAD. 
        /*
        int currMin = n; // highest height can be. we are looking for things less than this.
        
      
        
        vector<int> roots; 
        
        
        for(auto kv : m) {
            int i = kv.first;
                
            unordered_set<int> visited; 
            visited.insert(i);
            // Use set in method. Pass as reference!
            
            int h = this->getHeight(i, m, visited);
            // cout << "height for node " << i << " is " << h;
            if(h == currMin){
                roots.push_back(i);
            } else if(h < currMin) {
                
                currMin = h;
                roots.clear();
                roots.push_back(i);
            }
        }
        */
        
        // insert all leaf nodes into a queue, then slowly remove 
        // them from map and keep going until youre left with nodes that represent center
        
        queue<int> q; 
        queue<int> nextLevel; 
        
        for(auto kv: m){
            if(kv.second.size() == 1){
                cout << kv.first << endl;
                q.push(kv.first);
               //  cout << "LEAF IS : " << kv.first << endl;
            }
        }
        
        // OK SO ALGORITHM IS BASICALLY KEEP REMOVING LEAVES!
        // WHEN YOU GET TO CENTER, IF NEXT LEVEL QUEUE HAS 1 OR 2 ELEMENTS,
        // AND TOTAL ELEMENTS IS EITHER 1 OR 2, THEN YOU HAVE YOUR SOLUTION!
        // BASICALLY JUST PEEL AWAY LAYERS
  
        
        
        int counter = n; // Keep popping leaves!
        //THREE BASE CASES FOR MHT'S
        if(counter == 0) {
            return vector<int>{0};
        } else if(counter == 1) {
            return vector<int>{q.front()};
        } else if(counter == 2) {
            return vector<int>{q.back(), q.front()};
        }
        
            while(true){
                int head = q.front();
                q.pop();
                // cout << " POPPED THE FOLLOWING LEAF " << head << endl;
                
                --counter;
                
                for(auto child: m[head]){
                    
                    // delete leaf from child neighbor set.
                    // if child has set of size 1 afterwards, push em IN!
                    m[child].erase(head);
                    if(m[child].size() == 1) {
                        nextLevel.push(child);
                       
                    }
                    
                }
                m.erase(head);
                
                if(q.empty()) {
                    
                    /*
                    cout << " after popping outer leaves map looks liek this: " << endl;
                    
                    for(auto kv: m){
                        cout << "key " << kv.first;
                        for(auto v: kv.second) {
                            cout << " has val " << v << " " ;
                        }
                        cout << endl;
                    }
                   */
                    
                    if(counter == 1) {
                        return vector<int>{nextLevel.front()};
                    } else if(counter == 2) {
                        return vector<int>{nextLevel.front(), nextLevel.back()};
                    } else{
                        q = nextLevel; 
                        while(!nextLevel.empty())
                            nextLevel.pop();
                    }
                }
            
                   
            }
        // return roots; 
        //return vector<int>{1,2,3};
        }
        
       
    

    
};

// ################################################################################3

// ################################################################################3

// ################################################################################3
// FASTER SOLUTION 1



class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        if(n==1) return {0};
        else if(n==2) return {0,1};
        
        vector<int> adj[n];
        vector<int>deg(n,0);
        vector<int> res;
        
        for(auto e:edges) {
            deg[e[0]]++;
            deg[e[1]]++;
            adj[e[0]].push_back(e[1]);
            adj[e[1]].push_back(e[0]);
        }
        queue<int> leaf;
        int ctr=n;
        for(int i=0; i<n; i++) {
            if(adj[i].size()==1) {
                leaf.push(i);
            }
        }
        
        while(ctr>2) {
            for(int i=leaf.size(); i>0; i--){
                int l=leaf.front();
                //cout<<l<<" "<<deg[l]<<" "<<adj[l][0]<<" "<<deg[adj[l][0]]<<endl;;
                leaf.pop();
                deg[l]--;
                ctr--;
                for(int k=0; k<adj[l].size(); k++) {
                    if(--deg[adj[l][k]]==1) {
                        leaf.push(adj[l][k]);
                    }
                }
                
            }
        }
        while(!leaf.empty()) {
            res.push_back(leaf.front());
            leaf.pop();
        }
        return res;
    }
};


// EVEN FASTERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR

class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
        if (edges.empty()) return {0};
        
        vector<vector<int>> graph(n);
        for (auto& e : edges) {
            graph[e[0]].push_back(e[1]);
            graph[e[1]].push_back(e[0]);
        }
        vector<int> count(n), res;
        for (int i = 0; i < n; i++) count[i] = graph[i].size();
        
        queue<int> q;
        for (int i = 0; i < n; i++)
            if (count[i] == 1) q.push(i);
        // BFS
        while(!q.empty()) {
            res.clear();
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int leaf = q.front(); q.pop();
                res.push_back(leaf); count[leaf]--;
                for (int adj : graph[leaf]) {
                    if (count[adj] == 0) continue;
                    if (count[adj] == 2) q.push(adj);
                    count[adj]--;
                }
            }
        }
        return res;
    }
};


// SUPPER FAST SOLUTION


class Graph {
  int V;
  list<int>* adj;
    vector<int> degree;
  
    public:
    Graph(int V){
        this->V = V;
        adj = new list<int>[V];
        degree = vector<int>(V, 0);
    }
    
    void addEdge(int u, int v){
        adj[u].push_back(v);
        adj[v].push_back(u);
        degree[u]++;
        degree[v]++;
    }
    
    vector<int> findMinHeight(){
        if(V == 1){
            return vector<int>(1, 0);
        }
       queue<int> q;
        for(int i = 0; i < V; i++){
            if(degree[i] == 1)
                q.push(i);
        }
        
        while(V > 2){
            int size = q.size();
            for(int i = 0; i < size; i++){
                int u = q.front();
                q.pop();
                V--;
                
                for(auto it = adj[u].begin(); it != adj[u].end(); it++){
                    int v = *it;
                    degree[v]--; 
                    if (degree[v] == 1) 
                        q.push(v); 
                }
            }
        }
        vector<int> res; 
        while (!q.empty()) 
        { 
            res.push_back(q.front()); 
            q.pop(); 
        } 
        return res; 
    }
};
class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<pair<int, int>>& edges) {
        int min_height = INT_MAX;
        vector<int> result;
        
        Graph g(n);
        for(int i = 0; i < edges.size(); i++){
            g.addEdge(edges[i].first, edges[i].second);
        }
        
        return g.findMinHeight();
    }
};




// FASTEST SOLUTION

class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<pair<int, int>>& edges) {
        static int fast_io = []() { std::ios::sync_with_stdio(false); cin.tie(nullptr); return 0; }();
        vector<int> degree(n, 0);
        vector<list<int>> adjacent(n, list<int>());
        
        for(pair<int, int> edge : edges){
            ++degree[edge.first];
            ++degree[edge.second];
            adjacent[edge.first].push_back(edge.second);
            adjacent[edge.second].push_back(edge.first);
        }
        
        vector<int> leaf;
        int remain = n;
        for(int i = 0; i < n; ++i){
            if(degree[i] <= 1){
                leaf.push_back(i);
            }
        }
        
        while(remain > 2){
            vector<int> tem;
            for(int cur_leaf : leaf){
                for(int adj_leaf : adjacent[cur_leaf]){
                    --degree[adj_leaf];
                    if(degree[adj_leaf] == 1) tem.push_back(adj_leaf);
                }
            }
            remain -= leaf.size();
            leaf = tem;
        }
        
        return leaf;
    }
};
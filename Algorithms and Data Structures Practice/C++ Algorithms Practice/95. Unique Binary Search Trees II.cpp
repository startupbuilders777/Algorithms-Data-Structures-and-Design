/*

95. Unique Binary Search Trees II
Medium

1511

128

Favorite

Share
Given an integer n, generate all structurally unique BST's 
(binary search trees) that store values 1 ... n.

Example:

Input: 3
Output:
[
  [1,null,3,2],
  [3,2,null,1],
  [3,1,null,null,2],
  [2,1,3],
  [1,null,2,null,3]
]
Explanation:
The above output corresponds to the 5 unique BST's shown below:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3



*/
/*


The basic idea is that we can construct the result of n node tree just from the result of n-1 node tree.

Here's how we do it: only 2 conditions: 1) The nth node is the new root, so newroot->left = oldroot;

2) the nth node is not root, we traverse the old tree, every time the node in the old tree has a right child, 
we can perform: old node->right = nth node, nth node ->left = right child; and when we reach the 
end of the tree, don't forget we can also add the nth node here.

One thing to notice is that every time we push a TreeNode in our result, I push the clone version of the root, 
and I recover what I do to the old node immediately. 
This is because you may use the old tree for several times.

*/


// No memory leaks in this:


class Solution {
public:
    vector<TreeNode*> generateTrees(int n) {
        return generateTrees_1(n);
    }
    
    TreeNode * clone(TreeNode *old_root)
    {
        if(old_root == NULL) return NULL;
        TreeNode *new_root = new TreeNode(old_root->val);
        new_root->left = clone(old_root->left);
        new_root->right = clone(old_root->right);
        
        return new_root;
    }
    
    vector<TreeNode*> generateTrees_1(int n)
    {
        if(n <= 0) return {};
        vector<TreeNode *> results;
        vector<TreeNode *> previous_result(1, NULL);
        for(int i = 1; i <= n; ++i)
        {
            for(int j = 0; j < previous_result.size(); ++j)
            {
                // The nth node is the new root
                TreeNode *new_root = new TreeNode(i);
                TreeNode *new_left_subtree = clone(previous_result[j]);
                new_root->left = new_left_subtree;
                results.push_back(new_root);
                
                // traverse the old tree, use new node to replace the old right child
                TreeNode *root = previous_result[j];
                TreeNode *root_temp = root;
                while(root_temp != NULL)
                {
                    TreeNode *old_right_subtree = root_temp->right;
                    TreeNode *new_right_subtree = new TreeNode(i);
                    new_right_subtree->left = old_right_subtree;
                    root_temp->right = new_right_subtree;
                    TreeNode *new_tree = clone(root);
                    results.push_back(new_tree);
                    
                    root_temp->right = old_right_subtree;
                    delete new_right_subtree;
                    new_right_subtree = NULL;
                    root_temp = root_temp->right;
                }
            }
            
            swap(results, previous_result);
            results.clear();
        }
              
        return previous_result;
    }
};



// FASTEST SOLUTION POSSIBLE:

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    vector<TreeNode*> generateTrees(int n) {
        if(!n) return vector<TreeNode *> {};
        vector<int> v;
        for(int i=1;i<=n;i++) v.push_back(i);
        
        return f(v);
        
    }
    
    vector<TreeNode*> f(vector<int> v) {
        vector<TreeNode *> res;
        if(!v.size()) {
            res.push_back(NULL);
            return res;
        }
        
        for(int i = 0; i < v.size(); i++)
        {
            
            vector<int> lv(v.begin(), v.begin()+i);
            vector<int> rv(v.begin()+i+1, v.end());
            vector<TreeNode *> lres = f(lv);
            vector<TreeNode *> rres = f(rv);
            
            for(int j=0;j<lres.size();j++) {
                for(int k=0;k<rres.size();k++) {
                    TreeNode *node = new TreeNode(v[i]);
                    node->left = lres[j];
                    node->right = rres[k];
                    res.push_back(node);
                }
            }
            
        }
        return res;
        
    }
};



/*

Next Fastest:
*/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* deepCopyAdd(TreeNode* root,int add)
    {
        if(root==nullptr)
            return nullptr;
        TreeNode* t = new TreeNode(root->val+add);
        t->left=deepCopyAdd(root->left,add);
        t->right=deepCopyAdd(root->right,add);
        return t;
    }
    vector<TreeNode*> generateTrees(int n) {
        if(n==0)
            return vector<TreeNode*>();
        unordered_map<int,vector<TreeNode*>> n2t;
        n2t[0].push_back(nullptr);
        n2t[1].push_back(new TreeNode(1));
        TreeNode* nr=nullptr;
        for(int i=2;i<=n;i++)
        {
            for(int j=1;j<=i;j++)
            {
                for(auto &p1:n2t[j-1])
                {
                    for(auto &p2:n2t[i-j])
                    {
                        nr=new TreeNode(j);
                        nr->left=deepCopyAdd(p1,0);
                        nr->right=deepCopyAdd(p2,j);
                        n2t[i].push_back(nr);                        
                    }
                }

            }            
        }
        return n2t[n];
    }
};


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    
    vector<TreeNode*> genU(vector<int> v){
        
        if(v.size()==0)return {NULL};
        
        if(v.size()==1){
            TreeNode*x = new TreeNode(v[0]);
            return {x};
        }
        
        vector<TreeNode*> ans;
        vector<int> preRoot;
        vector<int> postRoot=v;
        postRoot.erase(postRoot.begin());
        for(int root=0;root<v.size();root++){
            vector<TreeNode*> pre = genU(preRoot);
            vector<TreeNode*> post = genU(postRoot);
            
            for(int i=0;i<pre.size();i++){
                for(int j=0;j<post.size();j++){
                   TreeNode*x = new TreeNode(v[root]);
                    x->left=pre[i];
                    x->right=post[j];
                    ans.push_back(x);
                }
            }
            preRoot.push_back(v[root]);
            if(!postRoot.empty())
            postRoot.erase(postRoot.begin());
        }
        return ans;     
    }
    
    
    
    
    vector<TreeNode*> generateTrees(int n) {
        vector<TreeNode*> l;
        if(n==0)return l;
        vector<int> v;
        for(int i=1;i<=n;i++)v.push_back(i);
        return genU(v);
    }
};

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    
    vector<TreeNode*> generateTreesFromRange(int start, int end) {
        cout << "start " << start << " end " <<  end << endl;
        
        if(start >= end){
            
            return vector<TreeNode*>({nullptr});
        }
        
        
        vector<TreeNode*> result;
        
        for(int i = start; i != end; ++i) {
                
                auto leftTrees = generateTreesFromRange(start, i);
                auto rightTrees = generateTreesFromRange(i+1, end);
                
                for(auto l: leftTrees ) {
                    
                    for(auto r: rightTrees) {
                        TreeNode * node = new TreeNode(i);
                        node->left = l;
                        node->right = r;
                        cout << "PUSHED" << endl;
                        result.push_back(node);
                    }
                }
        }
        
        return result;
    }
    
    
    vector<TreeNode*> generateTrees(int n) {
    
        // Better solution -> memozie these solutions, 
        // and deep copy instead of rebuilding these trees!
        if(n == 0) {
            
            return vector<TreeNode*>();
        
        }
        
        return generateTreesFromRange(1, n+1);
        
    }
};


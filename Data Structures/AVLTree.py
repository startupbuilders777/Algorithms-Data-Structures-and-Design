'''
AVL is a binary search tree such that
AVL -> Height balanced trees
AVL -> BST's where the difference between he height of the left and right subtrees is at most 1

height(R) - height(L) = 0 -> tree is balanced
height(R) - height(L) = -1 -> tree is left heavy
height(R) - height(L) = +1 -> tree is right heavy

BST Property -> left side has values less than root and right side has values greater than root

Insertion -> Do a standard BST insertion. Then recompute all the height(balance) expressions. If all the
expressions are still (-1, 0, 1) do nothing.
Otherwise Rotations

If the balance of a particular subtree is -2 (indicating left heavy),
    then a right rotation should be applied to that subtree
If the balance of a particular subtree is +2(indicating right heavy),
    then a left rotation should be applied to that subtree

Use an AVL Visual as I explain this:

FOR ALL THE ROTATIONS (SINGLE AND DOUBLE): You only apply at he node-level
                                           where the balance has been broken

In a rotation there is a parent, the left subtree and right subtree. The parent always becomes a
subtree and it is on the side with the smaller height. the side with the larger height becomes the new
root node(hence balancing the heights). The other side will remain with the parent as the parent descends
into a subtree.
The subtree and the root ascending to parent status will have a left and right side, call them X AND Y.
The parent will descend to the side with the smaller height. Since the parent lost a subtree (the subtree
that becomes the new parent), either X or Y will be the parents replacement left or right subtree.
The unchosen X or Y will remain with the ascended.

Here is a sort of idea of how it works:
-2 -> left heavy -> right rotation -> ascend the left subtree to parent status
+2 -> right heacy -> left rotation -> ascend the right subtree to parent status

Double rotations:
Done because the left or right side is too messed up(the heights for the left subtree are fcked, or the
heights of the right subtree are fcked so they need to be rotated before the parent node is rotated.

Double right rotation (inner left side is heavy -> fix that, then rotate entire tree):
 -> Left rotation of the left subtree, then a right rotation of the entire tree
 -> Can you also do right rotation of the left subtree, then right rotation on the entire tree?

Double left rotation(inner right side is heavy -> fix that, then rotate entire tree)
    -> Right rotatio

in a right rotation, the left subtree root is raised as the new root
'''

class Node():
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None

class avlTree():
    def __init__(self):
        self.root = None




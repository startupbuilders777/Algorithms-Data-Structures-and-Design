'''
AVL is a binary search tree such that
AVL -> Height balanced trees
AVL -> BST's where the difference between he height of the left and right subtrees is at most 1

height(R) - height(L) = 0 -> tree is balanced
height(R) - height(L) = -1 -> tree is left heavy
height(R) - height(L) = +1 -> tree is right heavy

'''

class Node():
    def __init__(self):
        self.data = None
        self.left = None
        self.right = None

class avlTree():
    def
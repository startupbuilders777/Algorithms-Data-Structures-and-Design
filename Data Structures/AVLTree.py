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

Single rotations work when the TREE IS:
LEFT LEFT Shape -> left side heavy -> Perform Right Rotation
RIGHT RIGHT SHAPE -> right side heavy -> Perform Left Rotation

Double Rotatations done when

LEFT RIGHT SHAPE -> left side heavy -> left rotate the subtree -> then right rotate entire tree
RIGHT LEFT SHAPE -> right side heavy -> right rotate the subtree -> then left rotate the entire tree

Double rotations:
Done because the left or right side is too messed up(the heights for the left subtree are fcked, or the
heights of the right subtree are fcked so they need to be rotated before the parent node is rotated.

Double right rotation (inner left side is heavy and the tree is left heavy -> fix that, then rotate entire tree):
 -> Left rotation of the left subtree, then a right rotation of the entire tree

Double left rotation(inner right side is heavy -and the tree is right heavy >fix that, then rotate entire tree)
-> Right rotation of the right subtree, then a left rotation of the entire tree

'''

class Node():
    def __init__(self):
        self.value = None
        self.height = 0
        self.left = None
        self.right = None

    def __init__(self, value):
        self.value = value
        self.height = 0
        self.left = None
        self.right = None

class avlTree():
    def __init__(self):
        self.root = None

    def balance(self):
        if(self.right is not None and self.left is not None):
            return self.right.height - self.left.height
        elif self.right is not None:
            return self.right.height
        elif self.left is not None:
            return self.left.height
        else:
            return 0


    '''
    other BST OPERATIONS
    FIND


    '''
    '''
    Going to hold off defining double right and double left rotation until i figure out
    why we cant just use single rotations all the time recursively.
    '''

    def rebalance(self):
        def rebalanceType(node):
            if (-1 < self.balance() and self.balance() < 1):
                return "NONE"
            elif (self.balance() <= -2):  # LEFT HEAVY
                return "RIGHT ROTATION"
            elif (self.balance() >= 2):  # RIGHT HEAVY
                return "LEFT ROTATION"

        def rightRotation(node):
            '''The left side is heavy'''
            newParent = node.left
            newRightSubtreeOfParent = node
            oldRightSubtreeOfNewParent = node.left.right
            node.left = oldRightSubtreeOfNewParent
            newParent.right = newRightSubtreeOfParent

        def leftRotation(node):
            '''The right side is heavy'''
            newParent = node.right
            newLeftSubtreeOfParent = node
            oldLeftSubtreeOfParent = node.right.left
            node.right = oldLeftSubtreeOfParent
            newParent.left = newLeftSubtreeOfParent
            
        def rebalanceRecursive(node, height):
            if(node.right is None and node.left is None):
                node.height = 1
                return 1

            node.left.height = rebalanceRecursive(node.left, height + 1)
            node.right.height = rebalanceRecursive(node.right, height + 1)

            type = rebalanceType(node)
            if(type == "RIGHT ROTATION"):
                rightRotation(no)

        heights = 0
        if(rebalanceType(self.root))


    def insert(self, value):
        def insertRecursive(node, value, height):
              if(node.value <= value): #left side contains values equal to the node value
                if(node.left is not None):
                    insertRecursive(node.left, value, height + 1)
                else:
                    node.left = Node(value)
                    node.right.height = height
              elif(node.value > value):
                if(node.right is not None):
                    insertRecursive(node.right, value, height + 1)
                else:
                    node.right = Node(value)
                    node.right.height = height
        if self.root is None:
            self.root = Node(value)
        else:
            insertRecursive(self.root, value, 0)




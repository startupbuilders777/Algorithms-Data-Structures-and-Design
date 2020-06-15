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
        self.parent = None

class AVLTree():
    def __init__(self):
        self.root = None

    @staticmethod
    def balance(node):
        if(node.right is not None and node.left is not None):
            return node.right.height - node.left.height
        elif node.right is not None:
            return node.right.height
        elif node.left is not None:
            return node.left.height
        else:
            return 0

    def rebalanceType(self, node):
        print("BALANCE VALUE IS: " + str(self.balance(node)))
        if (-1 <= self.balance(node) and self.balance(node) <= 1):
            return "NONE"
        elif (self.balance(node) <= -2):  # LEFT HEAVY
            return "RIGHT ROTATION"
        elif (self.balance(node) >= 2):  # RIGHT HEAVY
            return "LEFT ROTATION"
        else:
            print("EXCEPTION IN REBALANCE TYPE")

    '''
    Efficiency : T(n) = 2T(n/2) + O(1) = O(n)
    '''
    def checkIfAVLTreeIsBalanced(self):
        def checkIfAVLTreeIsBalancedRecursive(node):
            if node is None:
                return True
            else:
                left = node.left
                right = node.right
                balanceType = self.rebalanceType(node)

                if balanceType is "NONE":
                    leftBalanced = checkIfAVLTreeIsBalancedRecursive(left)
                    rightBalanced = checkIfAVLTreeIsBalancedRecursive(right)
                    if leftBalanced == True and rightBalanced == True:
                        return True
                    else:
                        return False

        return checkIfAVLTreeIsBalancedRecursive(self.root)

    '''
    other BST OPERATIONS
    FIND


    '''
    '''
    Going to hold off defining double right and double left rotation until i figure out
    why we cant just use single rotations all the time recursively.
    '''
    @staticmethod
    def printNode(node):
        print("VALUE OF NODE: " + str(node))
        print("HEIGHT IS: " + str(node.height))
        if(node.left is not None):
            print("VALUE OF NODE: " + str(node.left.value))
        elif(node.right is not None):
            print("VALUE OF NODE: " + str(node.right.value))


    def rebalance(self, node):
        def rightRotation(node):
            '''The left side is heavy'''
            print("RIGHT ROTATION STARTED")
            self.printNode(node)
            newParent = node.left
            newRightSubtreeOfParent = node
            oldRightSubtreeOfNewParent = node.left.right
            node.left = oldRightSubtreeOfNewParent
            newParent.right = newRightSubtreeOfParent
            self.printNode(newParent)
            # Put the rearranged nodes back into the tree
            if(node.parent is not None):
                if(node.parent.right == node):
                    node.parent.right = newParent
                elif(node.parent.left == node):
                    node.parent.left = newParent
            else:
                newParent.parent = None
                self.root = newParent

        def leftRotation(node):
            '''The right side is heavy'''
            print("LEFT ROTATION STARTED")
            self.printNode(node)
            newParent = node.right
            node.right.left.parent = node
            node.right = node.right.left
            newParent.left = node
            self.printNode(newParent)

            # Put the rearranged nodes back into the tree
            if (node.parent is not None):
                if (node.parent.right == node):
                    node.parent.right = newParent
                elif (node.parent.left == node):
                    node.parent.left = newParent
            else:
                newParent.parent = None
                self.root = newParent

            node.parent = newParent
        if(self.rebalanceType(node) == "NONE"):
            return
        elif(self.rebalanceType(node) == "LEFT ROTATION"):
            leftRotation(node)
        elif(self.rebalanceType(node) == "RIGHT ROTATION"):
            rightRotation(node)
        else:
            print("MESSED UP THE ROTATION")

    '''

    Following is the implementation for AVL Tree Insertion. 
    The following implementation uses the recursive BST insert to insert a new node. 
    In the recursive BST insert, after insertion, we get pointers to all ancestors 
    one by one in bottom up manner. So we don’t need parent pointer to travel up. 
    The recursive code itself travels up and visits all the ancestors of the newly inserted node.
    1) Perform the normal BST insertion.
    2) The current node must be one of the ancestors of the newly inserted node. Update the height of the current node.
    3) Get the balance factor (left subtree height – right subtree height) of the current node.
    4) If balance factor is greater than 1, then the current node is unbalanced and we are either in 
        Left Left case or left Right case. To check whether it is left left case or not, 
        compare the newly inserted key with the key in left subtree root.
    5) If balance factor is less than -1, then the current node is unbalanced and we are either in Right Right 
        case or Right Left case. To check whether it is Right Right case or not, compare the newly inserted key with the key in right subtree root.
    '''
    def insert(self, value):
        def insertRecursive(node, value, height):
            if(value < node.value):
                h = 0
                if(node.left is not None):
                    h = insertRecursive(node.left, value, height + 1)
                else:
                    node.left = Node(value)
                    node.left.height = 1
                    node.left.parent = node
                    h = height + 2
            elif(value >= node.value):
                if(node.right is not None):
                    h = insertRecursive(node.right, value, height + 1)
                else:
                    node.right = Node(value)
                    node.right.height = 1
                    node.right.parent = node
                    h = height + 2
            node.height = h - height

            '''Rebalance Ancestor Node after insertion'''
            self.rebalance(node=node)
            return h
        if self.root is None:
            self.root = Node(value)
        else:
            insertRecursive(self.root, value, 0)


tree = AVLTree()
tree.insert(2)
tree.insert(30)
tree.insert(4)
'''
tree.insert(12)
tree.insert(1)
tree.insert(2)
tree.insert(3)
tree.insert(4)
tree.insert(5)
'''
print(tree.checkIfAVLTreeIsBalanced())

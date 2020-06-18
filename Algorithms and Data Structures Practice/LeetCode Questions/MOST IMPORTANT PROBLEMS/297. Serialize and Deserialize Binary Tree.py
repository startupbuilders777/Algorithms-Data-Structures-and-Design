'''
297. Serialize and Deserialize Binary Tree
Hard

2864

141

Add to List

Share
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Example: 

You may serialize the following tree:

    1
   / \
  2   3
     / \
    4   5

as "[1,2,3,null,null,4,5]"
Clarification: The above format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

Note: Do not use class member/global/static variables to store states. Your serialize and deserialize algorithms should be stateless.
'''


# HARMANS SOLUNTION PREORDER
class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        # PREORDER ENCODING. 
        arr = []
        def encode(root):
            nonlocal arr 
            
            if(root is None):
                arr.append("#")
                return 
            # print("arr", arr)
            
            arr.append(root.val)
            encode(root.left)
            encode(root.right)
        
        encode(root)
        
        s = ""
        
        if(len(arr) > 0):
            s += str(arr[0])
        
        for i in range(1, len(arr)):
            s +=  " " +  str(arr[i]) 
        
        return s
        
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        arr = data.split()
        
        i = -1
        def build(arr):
            nonlocal i
            i += 1
            if i  >= len(arr): #.length:
                return None
            elif arr[i] == "#":
                return None
            n = TreeNode(arr[i])
            left = build(arr)
            right = build(arr)
            n.left = left
            n.right = right
            return n
        
        return build(arr)
    


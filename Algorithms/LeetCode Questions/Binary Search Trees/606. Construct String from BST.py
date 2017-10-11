'''
You need to construct a string consists of parenthesis and integers from a binary tree with the preorder traversing way.

The null node needs to be represented by empty parenthesis pair "()". And you need to omit all the empty parenthesis pairs that don't affect the one-to-one mapping relationship between the string and the original binary tree.

Example 1:
Input: Binary tree: [1,2,3,4]
       1
     /   \
    2     3
   /    
  4     

Output: "1(2(4))(3)"

Explanation: Originallay it needs to be "1(2(4)())(3()())", 
but you need to omit all the unnecessary empty parenthesis pairs. 
And it will be "1(2(4))(3)".
Example 2:
Input: Binary tree: [1,2,3,null,4]
       1
     /   \
    2     3
     \  
      4 

Output: "1(2()(4))(3)"

Explanation: Almost the same as the first example, 
except we can't omit the first parenthesis pair to break the one-to-one mapping relationship between the input and the output.


'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        result = [""]

        # print(type(t.val))
        # print(type(result))

        def tree2StrRecur(t, result):
            if (t is None):
                result = "()"
            else:
                if (t.left is not None and t.right is not None):
                    print(t.val, end="")
                    print("(", end="")
                    result[0] += str(t.val)
                    result[0] += "("
                    tree2StrRecur(t.left, result)
                    result[0] += ")"
                    result[0] += "("
                    print(")", end="")
                    print("(", end="")
                    tree2StrRecur(t.right, result)
                    print(")")
                    result[0] += ")"
                elif (t.left is None and t.right is not None):
                    result[0] += str(t.val) + "()" + "("
                    print(t.val, end="")
                    print("()", end="")
                    print("(", end="")
                    tree2StrRecur(t.right, result)
                    print(")", end="")
                    result[0] += ")"
                elif (t.right is None and t.left is not None):
                    result[0] += str(t.val)
                    result[0] += "("
                    print(t.val, end="")
                    print("(", end="")
                    tree2StrRecur(t.left, result)
                    print(")", end="")
                    result[0] += ")"
                else:
                    print(str(t.val), end="")
                    result[0] += str(t.val)

        tree2StrRecur(t, result)
        return result[0]


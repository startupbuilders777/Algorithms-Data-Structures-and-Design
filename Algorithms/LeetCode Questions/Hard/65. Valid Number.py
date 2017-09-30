'''
65. Valid Number
DescriptionHintsSubmissionsDiscussSolution
Discuss Pick One
Validate if a given string is numeric.

Some examples:
"0" => true
" 0.1 " => true
"abc" => false
"1 a" => false
"2e10" => true
Note: It is intended for the problem statement to be ambiguous. You should gather all requirements 
up front before implementing one.

Update (2015-02-10):
The signature of the C++ function had been updated. If you still see your function signature accepts a const char * argument, please click the reload button  to reset your code definition.

'''


class Solution(object):
    def isNumber(self, s):
        """
        :type s: str
        :rtype: bool
        """

        s = s.strip()  # Remove whitespace at front and end
        lst = [None] * len(s)

        for i in range(0, len(s)):
            if (s[i] == " "):
                return False
            elif (s[i].isdigit()):
                lst[i] = "d"
            elif (s[i] == "."):
                lst[i] = "p"
            elif (s[i] == "e"):
                lst[i] = "e"
        print(lst)

        if (lst.count("d") == 0):
            print("BAD 1")
            return False
        elif lst.count("p") > 1:
            print("BAD 2")
            return False
        elif lst[0] == "e":
            return False
        elif lst.count("e") > 1:
            print("BAD 3")
            return False
        elif len(lst) >= 1 and lst[-1] == "p" and lst.count("d") == 0:
            print("BAD 4")
            return False
        elif len(lst) >= 1 and lst[-1] == "e":
            print("BAD 5")
            return False
        else:
            return True
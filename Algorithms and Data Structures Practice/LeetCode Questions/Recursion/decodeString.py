'''
Given an encoded string, return its corresponding 
decoded string.

The encoding rule is: k[encoded_string], where 
the encoded_string inside the square brackets 
is repeated exactly k times. Note: k is guaranteed 
to be a positive integer.

Note that your solution should have linear 
complexity because this is what you will be 
asked during an interview.

Example

For s = "4[ab]", the output should be
decodeString(s) = "abababab";

For s = "2[b3[a]]", the output should be
decodeString(s) = "baaabaaa";

For s = "z1[y]zzz2[abc]", the output should be
decodeString(s) = "zyzzzabcabc".

'''


def decodeString(s):    
        stack = [("","")]
        currVal = ""
        for i in s:
            prevVal, prevRes = stack[-1]
            
            if i.isnumeric():
                currVal += i
            elif i.isalpha():
                stack.pop()
                prevRes += i
                stack.append((prevVal, prevRes))
            elif i == "[":
                stack.append((currVal, ""))
                currVal = ""
            elif i == "]":
                # just compute result?
                stack.pop() # remove stack frame
                result = prevRes
                if prevVal != "":
                    result = prevRes * int(prevVal)
                
                # frame before current one.
                lastVal, lastRes = stack.pop()
                lastRes += result
                stack.append((lastVal, lastRes))

        return stack[0][1]
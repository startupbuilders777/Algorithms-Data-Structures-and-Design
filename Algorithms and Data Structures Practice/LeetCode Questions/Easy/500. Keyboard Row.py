'''
Given a List of words, return the words that can be typed using letters of alphabet on only 
one row's of American keyboard like the image below.


American keyboard


Example 1:
Input: ["Hello", "Alaska", "Dad", "Peace"]
Output: ["Alaska", "Dad"]
Note:
You may use one character in the keyboard more than once.
You may assume the input string will only contain letters of alphabet.



'''


class Solution(object):
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        set1 = {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"}
        set2 = {"a", "s", "d", "f", "g", "h", "j", "k", "l"}
        set3 = {"z", "x", "c", "v", "b", "n", "m"}
        result = []
        for word in words:
            wordTemp = word.lower()
            useSet = None
            if (wordTemp[0] in set1):
                useSet = set1
            elif (wordTemp[0] in set2):
                useSet = set2
            elif (wordTemp[0] in set3):
                useSet = set3
            else:
                "error string is in no set"

            inSet = True

            for j in wordTemp:
                print(j)
                if j not in useSet:
                    inSet = False
                    break

            if (inSet):
                result.append(word)

        return result



'''

FASTER:

class Solution(object):
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        a=set('qwertyuiop');
        b=set('asdfghjkl');
        c=set('zxcvbnm');
        
        leng=len(words);
        result=[];
        for i in range(0,leng):
            if words[i][0].lower() in a:
                tmp=a;
            if words[i][0].lower() in b:
                tmp=b;
            if words[i][0].lower() in c:
                tmp=c;
            
            lengw=len(words[i]);
            flag=True
            for j in range(0,lengw):
                if words[i][j].lower() not in tmp:
                    flag=False
                    break
            if flag==True:
                result.append(words[i]);
        return result
            
            
'''


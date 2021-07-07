'''
1189. Maximum Number of Balloons
Easy

85

16

Favorite

Share
Given a string text, you want to use the characters of text to form as
many instances of the word "balloon" as possible.

You can use each character in text at most once. Return the maximum number 
of instances that can be formed.

 

Example 1:



Input: text = "nlaebolko"
Output: 1
Example 2:



Input: text = "loonbalxballpoon"
Output: 2
Example 3:

Input: text = "leetcode"
Output: 0
 

Constraints:

1 <= text.length <= 10^4
text consists of lower case English letters only.

'''

class Solution(object):
    def maxNumberOfBalloons(self, text):
        """
        :type text: str
        :rtype: int
        """
        
        
        
        '''
        just count all the letters in a map that are 'balloon'
        Then take max of counter map and return. 
        '''
        
        # This is a constant space solution because map only has 6 keys. 
        
        
        m = {'b': 0, 'a': 0, 'l': 0, 'o': 0, 'n': 0}
        
        
        for i in text:
            if m.get(i) is not None:
                m[i] += 1
        
        m['l'] /= 2
        m['o'] /= 2
        print(m)
        
        return min(m.values())
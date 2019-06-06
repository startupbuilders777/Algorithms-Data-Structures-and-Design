# Done
'''
Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

Example:

Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"
Note:

If there is no such window in S that covers all characters in T, return the empty string "".
If there is such window, you are guaranteed that there will always be only one unique minimum window in S.
'''



'''
Algorithm

We start with two pointers, leftleft and rightright initially pointing to the first element of the string SS.

We use the rightright pointer to expand the window until we get a desirable window i.e. a window that contains all of the characters of TT.

Once we have a window with all the characters, we can move the left pointer ahead one by one. If the window is still a desirable one we keep on updating the minimum window size.

If the window is not desirable any more, we repeat step \; 2step2 onwards.

'''

def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """

    if not t or not s:
        return ""

    # Dictionary which keeps a count of all the unique characters in t.
    dict_t = Counter(t)

    # Number of unique characters in t, which need to be present in the desired window.
    required = len(dict_t)

    # left and right pointer
    l, r = 0, 0

    # formed is used to keep track of how many unique characters in t are present in the current window in its desired frequency.
    # e.g. if t is "AABC" then the window must have two A's, one B and one C. Thus formed would be = 3 when all these conditions are met.
    formed = 0

    # Dictionary which keeps a count of all the unique characters in the current window.
    window_counts = {}

    # ans tuple of the form (window length, left, right)
    ans = float("inf"), None, None

    while r < len(s):

        # Add one character from the right to the window
        character = s[r]
        window_counts[character] = window_counts.get(character, 0) + 1

        # If the frequency of the current character added equals to the desired count in t then increment the formed count by 1.
        if character in dict_t and window_counts[character] == dict_t[character]:
            formed += 1

        # Try and contract the window till the point where it ceases to be 'desirable'.
        while l <= r and formed == required:
            character = s[l]

            # Save the smallest window until now.
            if r - l + 1 < ans[0]:
                ans = (r - l + 1, l, r)

            # The character at the position pointed by the `left` pointer is no longer a part of the window.
            window_counts[character] -= 1
            if character in dict_t and window_counts[character] < dict_t[character]:
                formed -= 1

            # Move the left pointer ahead, this would help to look for a new window.
            l += 1    

        # Keep expanding the window once we are done contracting.
        r += 1    
    return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]


'''
Approach 2: Optimized Sliding Window

Intuition

A small improvement to the above approach can reduce the time complexity of the algorithm to 
O(2*|filtered\_S| + |S| + |T|)O(2∗∣filtered_S∣+∣S∣+∣T∣), where filtered\_Sfiltered_S is the string 
    formed from S by removing all the elements not present in TT.

This complexity reduction is evident when |filtered\_S| <<< |S|∣filtered_S∣<<<∣S∣.

This kind of scenario might happen when length of string TT is way too small than the length of string SS and string SS consists of numerous characters which are not present in TT.

Algorithm

We create a list called filtered\_Sfiltered_S which has all the characters from string SS along with their indices in SS, but these characters should be present in TT.

  S = "ABCDDDDDDEEAFFBC" T = "ABC"
  filtered_S = [(0, 'A'), (1, 'B'), (2, 'C'), (11, 'A'), (14, 'B'), (15, 'C')]
  Here (0, 'A') means in string S character A is at index 0.
We can now follow our sliding window approach on the smaller string filtered\_Sfiltered_S.
'''

def minWindow(self, s, t):
    """
    :type s: str
    :type t: str
    :rtype: str
    """
    if not t or not s:
        return ""

    dict_t = Counter(t)

    required = len(dict_t)

    # Filter all the characters from s into a new list along with their index.
    # The filtering criteria is that the character should be present in t.
    filtered_s = []
    for i, char in enumerate(s):
        if char in dict_t:
            filtered_s.append((i, char))

    l, r = 0, 0
    formed = 0
    window_counts = {}

    ans = float("inf"), None, None

    # Look for the characters only in the filtered list instead of entire s. This helps to reduce our search.
    # Hence, we follow the sliding window approach on as small list.
    while r < len(filtered_s):
        character = filtered_s[r][1]
        window_counts[character] = window_counts.get(character, 0) + 1

        if window_counts[character] == dict_t[character]:
            formed += 1

        # If the current window has all the characters in desired frequencies i.e. t is present in the window
        while l <= r and formed == required:
            character = filtered_s[l][1]

            # Save the smallest window until now.
            end = filtered_s[r][0]
            start = filtered_s[l][0]
            if end - start + 1 < ans[0]:
                ans = (end - start + 1, start, end)

            window_counts[character] -= 1
            if window_counts[character] < dict_t[character]:
                formed -= 1
            l += 1    

        r += 1    
    return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]


from collections import OrderedDict

class Solution(object):
    def minWindow(self, s, t):
        need = collections.Counter(t)            #hash table to store char frequency
        missing = len(t)                         #total number of chars we care
        start, end = 0, 0
        i = 0
        for j, char in enumerate(s, 1):          #index j from 1
            if need[char] > 0:
                missing -= 1
            need[char] -= 1
            if missing == 0:                     #match all chars
                while i < j and need[s[i]] < 0:  #remove chars to find the real start
                    need[s[i]] += 1
                    i += 1
                need[s[i]] += 1                  #make sure the first appearing char satisfies need[char]>0
                missing += 1                     #we missed this first char, so add missing by 1
                if end == 0 or j-i < end-start:  #update window
                    start, end = i, j
                i += 1                           #update i to start+1 for next window
        return s[start:end]
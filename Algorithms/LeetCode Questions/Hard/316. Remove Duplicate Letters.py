'''
Given a string which contains only lowercase letters, remove duplicate letters so that every 
letter appear once and only once. You must make sure your result is the smallest in lexicographical 
order among all possible results.

Example:
Given "bcabc"
Return "abc"

Given "cbacdcbc"
Return "acdb"



'''


class Solution(object):
    def removeDuplicateLetters(self, s):
        """
        :type s: str
        :rtype: str
        """
        # Store tuple in map, letter and index letter was found,
        # then when you return, gutta return in the order of the tuples.
        # or dont store tuple, just store location as value for the key, which is a character
        def checkIfSmallerLetterExistsBeforeCurrentLetter(currentLetter, currentLetterValue, map):
            indexesOfAlphabet = alphabet = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8,
                                            "j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "q": 16,
                                            "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
                                            "z": 25}

            alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                        "t", "u", "v", "w", "x", "y", "z"]
            currentLetterIndex = indexesOfAlphabet[currentLetter]
            for i in alphabet[:currentLetterIndex]:
                if (map[i] <= currentLetterValue):
                    return True
            return False
        dict = {}

        index = 0  # index will always be below 26
        for i in range(0, len(s)):
            letter = s[i]
            if (dict.get(letter) is None):
                dict[letter] = index
                index += 1
            else:
                currentIndex = dict[letter]  # <- Reassign the currentIndex if a smaller letter exists before our index (linear search from a to z? slowdoe)                 if checkIfSmallerLetterExistsBeforeCurrentLetter(letter, currentIndex, dict):
                if checkIfSmallerLetterExistsBeforeCurrentLetter(letter, currentIndex, dict):
                    dict[letter] = index
                    index += 1
                else:
                    continue


        lst = [None] * 26
        for (letter, index) in dict.items():
            print(letter)
            print(index)
            lst[index] = letter

        result = ""
        for i in lst:
            if (i is not None):
                result += str(i)

        return result
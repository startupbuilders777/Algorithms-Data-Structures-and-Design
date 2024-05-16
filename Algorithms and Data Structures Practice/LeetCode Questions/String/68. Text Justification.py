"""
68. Text Justification
Solved
Hard
Topics
Companies
Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left-justified, and no extra space is inserted between words.

Note:

A word is defined as a character sequence consisting of non-space characters only.
Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array words contains at least one word.
 

Example 1:

Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
Example 2:

Input: words = ["What","must","be","acknowledgment","shall","be"], maxWidth = 16
Output:
[
  "What   must   be",
  "acknowledgment  ",
  "shall be        "
]
Explanation: Note that the last line is "shall be    " instead of "shall     be", because the last line must be left-justified instead of fully-justified.
Note that the second line is also left-justified because it contains only one word.
Example 3:

Input: words = ["Science","is","what","we","understand","well","enough","to","explain","to","a","computer.","Art","is","everything","else","we","do"], maxWidth = 20
Output:
[
  "Science  is  what we",
  "understand      well",
  "enough to explain to",
  "a  computer.  Art is",
  "everything  else  we",
  "do                  "
]
 

Constraints:

1 <= words.length <= 300
1 <= words[i].length <= 20
words[i] consists of only English letters and symbols.
1 <= maxWidth <= 100
words[i].length <= maxWidth


"""


class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        line = []
        word_len = 0 
        res = []
    
        for word in words:

            if len(word) + word_len + len(line) > maxWidth:
                spaces = maxWidth - word_len
                

                # watch out for case when there is only 1 word in the line..
                per_space_word = spaces // ((len(line) - 1) if len(line) > 1 else 1)
                remain_space = spaces % ((len(line) - 1) if len(line) > 1 else 1)

                res_line = ""
                for idx, i in enumerate(line):

                    # ADD WORD .. then add space.. -> end in word though!
                    # unless there is only 1 word!
                    
                    res_line += i
                    if idx < len(line) - 1 or len(line) == 1:
                        res_line += " "*per_space_word 
                        res_line += " " if remain_space > 0 else ""
                        remain_space -= 1

                res.append(res_line)
                line = [word]
                word_len = len(word) 
            else:
                line.append(word)
                word_len += len(word) 

        # LEFT JUSTIFY THIS
        if len(line) > 0:
                def left_justify(line):
                    res_line = " ".join(line)
                    spaces = maxWidth - word_len - (len(line) -1 )
                    res_line += " "*spaces
             
                    res.append(res_line)

                left_justify(line)

        return res
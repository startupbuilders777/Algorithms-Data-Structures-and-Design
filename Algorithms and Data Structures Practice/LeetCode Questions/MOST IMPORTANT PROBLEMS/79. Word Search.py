'''
79. Word Search
Medium

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially 
adjacent cell, where "adjacent" cells are those horizontally
or vertically neighboring. The same letter cell may not be used more than once.

Example:

board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.


'''


class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        
        R = len(board)
        C = len(board[0])
        
        def dfs(word, idx, row, col, visited):
            
            if((row, col) in visited):
                return False
            
            else:
                visited.add((row, col))
            
            
            if(idx == len(word)):
                return True
            
            if(row + 1 < R and board[row + 1][col] == word[idx] ):
                up = dfs(word, idx+1, row+1, col, visited)
                # THIS IS PRUNING BELOW. RETURN FAST!
                if up: 
                    return True
            
            if(row - 1 >= 0 and board[row-1][col] == word[idx]):
                down = dfs(word, idx+1, row-1, col, visited)
                if down:
                    return True
            
            if(col + 1 < C and board[row][col+1] == word[idx]):
                right = dfs(word, idx+1, row, col+1, visited)
                if right: 
                    return True
                
            
            if(col - 1 >= 0 and board[row][col-1] == word[idx]):
                left = dfs(word, idx+1, row, col-1, visited)
                if left: 
                    return True
            
            # By doing this, we can reuse set for all brute forcing,
            # because it is automatically emptied. 
            
            # Dont aggregrate, return True asap. DO PRUNING, AND NO 
            # UNNCESSARY SEARCH!
            visited.remove((row, col))
            # return any([up, down, left, right])
            return False
        
        # We are going to reuse set structure!
        
        # Can you memoize failure?
        # m[idx, row, col] stores whether suffix of word was found at that location. 
        
        visited = set()        
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                
                if(board[i][j] == word[0] and dfs(word, 1, i, j, visited)):
                    return True
        
        return False
    
'''
FASTEST:

TO ACHIEVE CONSTANT SPACE,
DONT USE VISTIED SET, USED GRID ITSELF!!
'''

class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
	
        def CheckLetter(row, col, cur_word):
		#only the last letter remains
            if len(cur_word) == 1:
                return self.Board[row][col] == cur_word[0]
            else:
			#mark the cur pos as explored -- None so that other can move here
                self.Board[row][col] = None
                if row+1<self.max_row and self.Board[row+1][col] == cur_word[1]:
                    if CheckLetter(row+1, col, cur_word[1:]):
                        return True
                if row-1>=0 and self.Board[row-1][col] == cur_word[1]:
                    if CheckLetter(row-1, col, cur_word[1:]):
                        return True
                if col+1<self.max_col and self.Board[row][col+1] == cur_word[1]:
                    if CheckLetter(row, col+1, cur_word[1:]):
                        return True
                if col-1>=0 and self.Board[row][col-1] == cur_word[1]:
                    if CheckLetter(row, col-1, cur_word[1:]):
                        return True
				#revert changes made
                self.Board[row][col] = cur_word[0]
                return False                  
       
        self.Board = board
        self.max_row = len(board)
        self.max_col = len(board[0])
        if len(word)>self.max_row*self.max_col:
            return False
        for i in range(self.max_row):
            for j in range(self.max_col):
                if self.Board[i][j] == word[0]:
                    if CheckLetter(i, j, word):return True
        return False
        
                
                
                
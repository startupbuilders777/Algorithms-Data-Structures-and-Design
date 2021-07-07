'''
1105. Filling Bookcase Shelves
Medium

574

30

Add to List

Share
We have a sequence of books: the i-th book has thickness books[i][0] and height books[i][1].

We want to place these books in order onto bookcase shelves that have total width shelf_width.

We choose some of the books to place on this shelf (such that the sum of their thickness is <= shelf_width), then build another level of shelf of the bookcase so that the total height of the bookcase has increased by the maximum height of the books we just put down.  We repeat this process until there are no more books to place.

Note again that at each step of the above process, the order of the books we place is the same order as the given sequence of books.  For example, if we have an ordered list of 5 books, we might place the first and second book onto the first shelf, the third book on the second shelf, and the fourth and fifth book on the last shelf.

Return the minimum possible height that the total bookshelf can be after placing shelves in this manner.

 

Example 1:


Input: books = [[1,1],[2,3],[2,3],[1,1],[1,1],[1,1],[1,2]], shelf_width = 4
Output: 6
Explanation:
The sum of the heights of the 3 shelves are 1 + 3 + 2 = 6.
Notice that book number 2 does not have to be on the first shelf.
 

Constraints:

1 <= books.length <= 1000
1 <= books[i][0] <= shelf_width <= 1000
1 <= books[i][1] <= 1000
'''

class Solution:
    def minHeightShelvesTopDown(self, books: List[List[int]], shelf_width: int) -> int:
        '''
        put as many books within current shelf width as possible
        then build new shelf width?
        
        maybe not, you can choose a bigger book. 
        for the next shelf. 
        
        if its the same height put it on the same shelf width. 
        if its bigger, then... 
            
            depends how much bigger. 
        
        HAVE TO DO DP TO FIGURE IT OUT. CANT GREEDY THIS. 
        but if books is same size or smaller, keep on same shelf. 
        '''
        
        # Top down
        @lru_cache(maxsize=None)
        def dp(i, currHeight, currWidth):
            
            if i == len(books):
                return currHeight
            
            # current height?
            
            thickness, height = books[i] 
            if thickness > currWidth:
                return currHeight + dp(i + 1, height, shelf_width-thickness)
            elif height <= currHeight:
                return dp(i+1, currHeight, currWidth-thickness)
            else:                
                # put in this shelf, or create new one, take min. 
                thisShelf = dp(i + 1, height, currWidth - thickness)
                nextShelf = currHeight + dp(i+1, height, shelf_width-thickness)
                
                return min(thisShelf, nextShelf)
        
        return dp(0, 0, shelf_width)

    '''
    So dp[k] means standing at index k, 
    what is the best total height I can get.
    With that being said, translate 
    dp[i] = min(dp[i], dp[j] + max_height 
    
    in English is:
    My current best solution is to either keep the recent 
    books in one last row from previous walkthrough dp[i], 
    OR, make dp[j] as all previous best solution + having 
    some books with max_height with me in the same row.
    '''
    
    def minHeightShelves(self, books: List[List[int]], shelf_width: int) -> int:
        n = len(books)
        dp = [float('inf') for _ in range(n + 1)]
        dp[0] = 0
        for i in range(1, n + 1):
            max_width = shelf_width
            max_height = 0
            j = i - 1
            while j >= 0 and max_width - books[j][0] >= 0:
                max_width -= books[j][0]
                max_height = max(max_height, books[j][1])
                dp[i] = min(dp[i], dp[j] + max_height)
                j -= 1
        return dp[n]
'''
Each cell in a 2D grid contains either a wall ('W') or an enemy ('E'), or is empty ('0'). Bombs can destroy enemies, but walls are too strong to be destroyed. A bomb placed in an empty cell destroys all enemies in the same row and column, but the destruction stops once it hits a wall.

Return the maximum number of enemies you can destroy using one bomb.

Note that your solution should have O(field.length · field[0].length) 
complexity because this is what you will be asked during an interview.

Example

For

field = [["0", "0", "E", "0"],
         ["W", "0", "W", "E"],
         ["0", "E", "0", "W"],
         ["0", "W", "0", "E"]]
the output should be
bomber(field) = 2.

Placing a bomb at (0, 1) or at (0, 3) destroys 2 enemies.

Input/Output

[execution time limit] 4 seconds (py3)

[input] array.array.char field

A rectangular matrix containing characters '0', 'W' and 'E'.

Guaranteed constraints:
0 ≤ field.length ≤ 100,
0 ≤ field[i].length ≤ 100.

[output] integer

[Python 3] Syntax Tips

PRECOMPUTED SOLUTION BELOW:
'''

def bomber(field):
    
    '''
    count bombs to the right. for each index()
    save the max everywhere. 
    count bombs below for each index?
    can do with DP and recursion -> count max size to the right. save it. 
    count max size down? save it ??
    keep track of max count. 
    
    lr[][] -> left and right precomputed result. 
    td[][] -> top down precomputed result. 
    '''    
    
    R = len(field)
    
    if R == 0:
        return 0
    
    C = len(field[0])
    
    td = [[0 for i in range(C)] for j in range(R)]
    lr = [[0 for i in range(C)] for j in range(R)]
    
    # go left to right and fille lr !
    for i in range(R):
        start = 0
        cnt = 0
        for j in range(C):
            if field[i][j] == "E":
                cnt += 1
            elif field[i][j] == "W":
                # add enemeies into precomputed         
                for t in range(start, j):
                    if field[i][t] == "E" or field[i][t] == "0":
                        lr[i][t] = cnt         
                cnt = 0
                start = j 
        # run to the end? 
        for t in range(start, C):
            if field[i][t] == "E" or field[i][t] == "0":
                lr[i][t] = cnt      
 
    
    for j in range(C):
        start = 0
        cnt = 0
        for i in range(R):
            if field[i][j] == "E":
                cnt += 1
            elif field[i][j] == "W":
                # add enemeies into precomputed         
                for t in range(start, i):
                    if field[t][j] == "E" or field[t][j] == "0":
                        td[t][j] = cnt         
                cnt = 0
                start = i
                
        # run to the end? 
        for t in range(start, R):
            if field[t][j] == "E" or field[t][j] == "0":
                td[t][j] = cnt      
    
    m = 0
    
    for i in range(R):
        for j in range(C):
            # BOMBS CAN ONLY BE PLACED ON EMPTY CELLS!
            if field[i][j] == "0":
                m = max(m, lr[i][j] + td[i][j])
    return m
            
'''
C++ PRE compute soln:


int bomber(vector<vector<char>> f) {
    int x = f.size();
    if (x == 0)
        return 0;
    int y = f[0].size();
    vector<vector<int>> kill(x,vector<int>(y,0));
    for (int i = 0; i < x; ++i) {
        int j_min = 0;
        int num_e = 0;
        for (int j = 0; j <= y; ++j) {
            if (j == y || f[i][j] == 'W') {
                for (; j_min < j; ++j_min)
                    if (f[i][j_min] == '0')
                        kill[i][j_min] += num_e;
                ++j_min;
                num_e = 0;
            } else if (f[i][j] == 'E') {
                ++num_e;
            }
        }
    }
    for (int j = 0; j < y; ++j) {
        int i_min = 0;
        int num_e = 0;
        for (int i = 0; i <= x; ++i) {
            if (i == x || f[i][j] == 'W') {
                for (; i_min < i; ++i_min)
                    if (f[i_min][j] == '0')
                        kill[i_min][j] += num_e;
                ++i_min;
                num_e = 0;
            } else if (f[i][j] == 'E') {
                ++num_e;
            }
        }
    }
    int best = 0;
    for (r : kill)
        for (e : r)
            best = max(best,e);
    return best;
}


'''

# CAN ALSO DO THIS WAY: 
# Hint: DP O(nm) TOP/LEFT + DOWN/RIGHT



def bomber(A):
    from itertools import groupby
    if not A or not A[0]: return 0
    R, C = len(A), len(A[0])
    dp = [ [0] * C for _ in xrange(R) ]
    for r, row in enumerate(A):
        c = 0
        for k, v in groupby(row, key = lambda x: x != 'W'):
            w = list(v)
            if k:
                enemies = w.count('E')
                for c2 in xrange(c, c + len(w)):
                    dp[r][c2] += enemies
            c += len(w)

    for c, col in enumerate(zip(*A)):
        r = 0
        for k, v in groupby(col, key = lambda x: x != 'W'):
            w = list(v)
            if k:
                enemies = w.count('E')
                for r2 in xrange(r, r + len(w)):
                    dp[r2][c] += enemies
            r += len(w)
    
    ans = 0
    for r, row in enumerate(A):
        for c, val in enumerate(row):
            if val == '0':
                ans = max(ans, dp[r][c])
    return ans



def bomber(F):
    if not F or not F[0]         :   return 0
    row ,col = len(F) ,len(F[0]) ;   F = numpy.array(F)
    dp = numpy.zeros((row,col))  ;   t = zip(*numpy.where(F == 'E'))
    for x,y in t:
        for i in range(y-1,-1,-1):   
            if F[x,i] == 'W'  :   break
            if F[x,i] == '0' :   dp[x,i]+=1 
        for i in range(y+1,col):
            if F[x,i] == 'W'  :   break
            if F[x,i] == '0'  :   dp[x,i]+=1 
        for i in range(x-1,-1,-1):
            if F[i,y] == 'W'  :   break
            if F[i,y] == '0'  :   dp[i,y]+=1 
        for i in range(x+1,row):
            if F[i,y] == 'W'  :   break
            if F[i,y] == '0'  :   dp[i,y]+=1 
    return dp.max()
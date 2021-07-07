'''
    1) CONNECT THE CITIES

    Problem Statement
            
    A and B are two cities distance units away from each other. 
    Several transmitters have been placed along the straight road connecting them. 
    The transmission range can be set to any positive integer value, but it must be 
    the same for all transmitters. Any two transmitters can communicate directly 
    if the distance between them is not greater than the transmission range. Each 
    transmitter can communicate with city A or city B if the distance between the 
    transmitter and the city is not greater than the transmission range.

    You have been assigned to set up a connection between the cities. 
    You are allowed to move any number of transmitters, but moving a 
    transmitter for k units costs you k dollars and the budget does 
    not allow you to spend more than funds dollars in total. You can 
    move the transmitters into points with integer coordinates only.

    You will be given a int[] position, with the i-th element of position 
    representing the initial distance between the i-th transmitter and city A. 
    You will be also given funds, the maximal total cost you are allowed to spend 
    when moving transmitters. Return the minimal transmission range which still 
    allows you to establish a connection between the cities. See notes 
    for the formal definition of the connection.

    Definition
            
    Class:	ConnectTheCities
    Method:	minimalRange
    Parameters:	int, int, int[]
    Returns:	int
    Method signature:	int minimalRange(int distance, int funds, int[] position)
    (be sure your method is public)
        
    
    Notes
    -	Cities A and B are connected if there exists a sequence of transmitters t1, t2, ..., tn 
        such that transmitter t1 can communicate directly with city A, transmitter tn can communicate 
        directly with city B and, for every i between 1 and n-1, inclusive, transmitters ti and ti+1 
        can communicate directly.
    
    Constraints
    -	distance will be between 1 and 100, inclusive.
    -	position will contain between 1 and 50 elements, inclusive.
    -	Each element of position will be between 0 and distance, inclusive.
    -	funds will be between 0 and 100000, inclusive.
    
    Examples

    0)		
    10
    5
    { 3, 7, 9 }
    Returns: 3
    We can move the second transmitter one unit left and the range of 3 units will be enough for a connection.

    1)	   	
    20
    100
    { 0, 0, 0, 0 }
    Returns: 4
    We have enough money to place transmitters at positions 4, 8, 12 and 16.

    2)	   	
    63
    19
    { 34, 48, 19, 61, 24 }
    Returns: 12
'''
'''
    HARMAN SOLUTION:

    We have to set transmission range, has to be less 
    than distance between 2 citiies.    
    
    We are finding min transmission range. 
    We have the distance between A and B. 

    We place the transmitters, in the same way as they are given to us in position 
'''
    
def minimalRange(distance, funds, position):
    '''
    Backtrack soln, place them if you exceed funds, stop and try a differnt 
    way enumerate all possibilties, take minimum one? 
    The next i has to start from the current i!
    we want to place them as close to each other at the beginning, then increase the range. 
    but also cant go outside budget. 
    '''
    
    # TOP DOWN?
    def solveTD(i, last_position, totalCost):
        if totalCost >= funds:
            return False, -1
        
        if i == len(position) - 1:
            # is i close enough to tower?
            return (True, distance - last_position)
        
        
        
        placement_len = distance/len(position)
        # try all distances, from 1 to (distance/number of transmitters)
        # last_position + 1 to last_position + 
        
        min_l = float("inf")
        
        for dist in range(last_position + 1, last_position + placement_len):
        
            possible, l = solve(i+1, dist, totalCost + abs(position[i] -  dist) ) 
            
            if possible != False:
                min_l = min(min_l, dist, l)
        
        return min_l

    return solve(0, 0, 0)

'''
ConnectTheCities

According to the problem statement, each transmitter can be moved to any location 
without constraints. But it is useless to change order of transmitters. It can be 
proven as follows: if you have solution where two transmitters change order, swap 
their final destinations and number of moves won't increase for sure but the connectivity 
will remain the same. So we can assume that in optimal solution transmitters are placed 
in the same order as they stay initially.

In DP solution we place transmitters one-by-one from left to right. 
Besides number of transmitters already placed, we need to store the position 
of foremost transmitter. This information is necessary to check connectivity of 
our chain. Also not to exceed move limit we have to include number of moves made 
so far into the state of DP. The state domain is (k,f,m)->r where k is number of 
already placed leftmost transmitters, f is position of front transmitter, 
m is number of moves made and r is minimal transmittion range necessary to 
maintain connectivity. Transition is performed by setting the k-th transmitter 
to some p coordinate which is greater or equal to f. This transmitter becomes foremost one, 
transmission range can be increased to (p — f) if necessary and number of 
moves increases by |p — X[k]|. The problem answer is produced by considering 
all possible placements of all n transmitters and trying to connect the
last one to the city B. This DP solution is O(N * D^2 * M) where N is number of transmitters, 
D is distance between cities and M is maximum number of moves allowed.


It is easy to see that if the number of moves m done so far increases then the 
partial solution worsens because the remaining limit on moves decreases and 
the set of possible continuations narrows. So we can rotate the DP 
problem with parameters m and r. The new state domain is (k,f,r)->m where 
r is exact transmitter range required for connectivity and m is minimal number 
of used moves possible with such parameters. The transition as an operation 
on (k,f,r,m)-tuple remains the exactly same. When we calculate the problem answer 
we do the same things but also we check that the number of moves 
does not exceed the limit. The rotated DP is O(N * D^3) in time and can be 
done in O(D^2) space complexity if "store two layers" optimization is used.
So basically the O(N * D^2 * M) became O(N * D^2 * D), and this is efficient if D < M. 


int n, d;                                           //number of transmitters and distance between cities
int res[2][MAXD][MAXD];                             //state domain results: (k,f,r)->m
...
    sort(xarr.begin(), xarr.end());                 //do not forget to sort the transmitter positions!
    memset(res, 63, sizeof(res));
    res[0][0][0] = 0;                               //DP base: all zeros possible, others impossible
    for (int k = 0; k<n; k++) {                     //iterate k  -  number of transmitters places so far
      memset(res[(k+1)&1], 63, sizeof(res[0]));
      for (int f = 0; f<=d; f++)                    //iterate f  -  position of foremost(and last) transmitter
        for (int r = 0; r<=d; r++) {                //iterate r  -  required transmission range
          int m = res[k&1][f][r];                   //get minimal number of moves required
          if (m > maxmoves) continue;               //discard state if it is impossible or number of moves exceeds the limit
          for (int p = f; p<=d; p++)                //iterate p  -  possible position of k-th transmitter
            relax(res[(k+1)&1][p][max(r,p-f)], m + abs(p-xarr[k])); //try transition
        }
    }
    int answer = 1000000000;                        //getting answer as minimal possible r
    for (int f = 0; f<=d; f++)                      //over all states with k = n
      for (int r = 0; r<=d; r++) {
        int m = res[n&1][f][r];
        if (m > maxmoves) continue;                 //with number of moves under limit
        int rans = max(r, d-f);                     //do not forget the last segment for transmission
        relax(answer, rans);
      }
    return answer;



'''

    
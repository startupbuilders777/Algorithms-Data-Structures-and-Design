'''
 Problem Statement for RoadOrFlightHard


Problem Statement
     The king has been out to work for a long time and he wants to go back to 
     his queen as fast as possible. The king is in city 0 and the queen is in city N. 
     There are roads and flights connecting city i to city (i+1) for all i between 
     0 and N-1, inclusive.

The time it takes to travel from city i to city (i+1) by road and by flight is given by the 
i-th elements of roadTime and flightTime, respectively. roadTime can be generated from the following pseudocode:

    roadTime[0] = roadFirst mod roadMod;
    for i = 1 to N-1
        roadTime[i] = (roadTime[i-1]*roadProd + roadAdd) mod roadMod;
        // note that (roadTime[i-1]*roadProd + roadAdd) may overflow a 32-bit integer
        
flightTime can be generated similarly by using flightFirst, flightProd, flightAdd, flightMod.



However, taking a flight risks the life of the king during takeoffs 
due to the technological limitations in the kingdom. Hence the queen 
has asked him to ensure that the total number of takeoffs during his 
entire journey does not exceed K.

To minimize the number of takeoffs, the king may choose to take a direct 
flight from city i to city i+j instead of separate flights from city i to 
city i+1, then from i+1 to i+2, ... and from i+(j-1) to i+j. The time taken 
for this flight is the sum of the times taken for the flights from i to i+1, i+1 to i+2, ..., i+(j-1) to i+j.


Return the minimum amount of time in which the king can reach his queen.

 
Definition
    	
Class:	RoadOrFlightHard
Method:	minTime
Parameters:	int, int, int, int, int, int, int, int, int, int
Returns:	long
Method signature:	long minTime(int N, int roadFirst, int roadProd, int roadAdd, int roadMod, 
                                 int flightFirst, int flightProd, int flightAdd, int flightMod, int K)
(be sure your method is public)
    
 
Constraints
-	N will be between 1 and 400000, inclusive.
-	roadFirst, roadAdd, flightFirst and flightAdd will each be between 0 and 100000, inclusive.
-	roadProd, roadMod, flightProd and flightMod will each be between 1 and 100000, inclusive.
-	K will be between 0 and 40, inclusive.
-	K will be less than or equal to N.
 
Examples
0)	
    	
3
14
1
2
10
18
1
10
17
1
Returns: 14
The pseudocode gives roadTime = {4, 6, 8} and flightTime = {1, 11, 4}. 
The fastest way to reach the queen is to take the road from city 0 to 1 and 1 to 2, and a flight from city 2 to 3.

1)	
    	
3
4
1
2
10
1
1
10
17
2
Returns: 11
roadTime and flightTime are the same as in previous example. But now the king is allowed 2 takeoffs.
2)	
    	
3
4
1
2
10
1
1
6
9
1
Returns: 12
roadTime = {4, 6, 8} and flightTime = {1, 7, 4}. Even though roadTime[1] < flightTime[1], 
it is best to take a direct flight from city 0 to city 3 which takes a total time of 1 + 7 + 4 = 12 units.
3)	
    	
5
85739
94847
93893
98392
92840
93802
93830
92790
3
Returns: 122365

This problem statement is the exclusive and proprietary property of TopCoder, 
Inc. Any unauthorized use or reproduction of this information without the 
prior written consent of TopCoder, Inc. is strictly prohibited. (c)2010, 
TopCoder, Inc. All rights reserved.

'''

'''
Harmans answer:


roads
flights

DP(i, k) -> totalTime

i processes the town/road we on!

roadArray -> create from the mod thing above
flightArray -> create from mod thing above

'''


'''
In this problem we are asked to find minimal distance between two points 
in a special graph (with line structure). Just like in Dijkstra algorithm, 
state domain should include v parameter — city number and the result of state 
is minimal distance from starting city (0) to current city (v). Also there is 
constraint on number of takeoffs, so we have to memorize that number as parameter t. 

After these obvious thoughts we have the state domain (v,t)->D, where v=0..n and t=0..k. 
Unlike previous examples, this DP solution follows backwards(recurrent) style, so each 
result is determined using the previous results. The transition rules are rather simple. 

If king comes from previous city by ground, then D(v,t) = D(v-1,t) + R[v-1] where 
R is array of road travelling times. If king arrives to current city by plane from city u (u < V), 
then D(v,t) = D(u,t-1) + (F[u] + F[u+1] + F[u+2] + ... + F[v-2] + F[v-1]). 

Since any of such conditions may happen, the result of minimum of results 
for road and flight cases for all possible departure cities. 
The problem answer is min(D(n,t)) for t=0..k.

Unfortunately, this DP has O(N^3 * K) time complexity. For each possible 
flight the sum of array elements in a segment is calculated and this calculation 
results in the innermost loop. To eliminate this loop we have to precalculate 
prefix sums for flight times. Let S(v) = F(0) + F(1) + F(2) + ... + F(v-1) 
for any v=0..n. This array is called prefix sums array of F and it can be 
computed in linear time by obvious DP which originates from these recurrent 
equations: S(0) = 0; S(v+1) = S(v) + F(v). Having prefix sums array, 
sum of elements of any segment can be calculated in O(1) time because F(L) + ... + F(R-1) = S(R) — S(L). 

So the recurrent relations for flight case are rewritten as: D(v,t) = D(u,t-1) + (S(v) — S(u)). 
The time complexity is now O(N^2 * K).

The next optimization is not that easy. The best result in case king arrives to 
city v by some plane is D(v,t) = min_u(D(u,t-1) + S(v) — S(u)) where u takes values 

from 0 to v-1. Transform the equation: 

D(v,t) = min_u(D(u,t-1) — S(u) + S(v)) = min_u(D(u,t-1) — S(u)) + S(v). 

Notice that the expression inside the minimum does not depend on v anymore. 
Let's denote the whole minimum as A(v-1,t-1). The A array is added to state domain 
and its contents can be calculated during the DP. It is interesting to discuss the 
meaning of A(v,t). I would say it is the best virtual flight distance from city 0 to city v. 

It is virtual because the flight can start at any city. Here are the full and final transitions: 
D(v,t) = min(D(v-1,t) + R[v-1], A(v-1,t-1) + S(v)); A(v,t) = min(A(v-1,t), D(v,t) — S(v));

Now the solution works in O(N * K) time. But the size of results arrays 
exceed memory limit. The workaround is very simple. Notice that the final DP is 
layered by v parameter because to get results (v,t) only (v-1,*) and (v,*) 
states are required. So we can store results only for two adjacent layers at any time.
After this optimization space complexity becomes linear O(N + K).

int64 sum[MAXN];                                              //prefix sums array S
int64 vfd[2][MAXK];                                           //A(v,t)  -  virtual flight distance
int64 res[2][MAXK];                                           //D(v,t)  -  minimal distance to city v with t takeoffs
...
  sum[0] = 0;
  for (int v = 0; v<n; v++) sum[v+1] = sum[v] + flight[v];    //prefix sums calculation
  
  memset(res, 63, sizeof(res));                               //DP base for city 0:
  memset(vfd, 63, sizeof(vfd));                               //only zero takeoffs with zero distance
  res[0][0] = 0;                                              //all other entries are infinities
  vfd[0][0] = 0;
  for (int v = 1; v<=n; v++)                                  //iterate through all states with v>0
    for (int t = 0; t<=k; t++) {                              
      res[v&1][t] = res[(v-1)&1][t] + road[v-1];              //try road way to here
      if (t > 0) res[v&1][t] = min(res[v&1][t], vfd[(v-1)&1][t-1] + sum[v]); //try flight arrival here
      vfd[v&1][t] = min(vfd[(v-1)&1][t], res[v&1][t]  -  sum[v]); //try flight departure here
    }
  int64 answer = INF;                                         //find minimal distance to city n
  for (int t = 0; t<=k; t++) answer = min(answer, res[n&1][t]);
  return answer;


'''
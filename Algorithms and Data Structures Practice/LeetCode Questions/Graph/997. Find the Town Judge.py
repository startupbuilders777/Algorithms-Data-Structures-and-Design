'''
997. Find the Town Judge
Easy

752

83

Add to List

Share
In a town, there are N people labelled from 1 to N.  
There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:

The town judge trusts nobody.
Everybody (except for the town judge) trusts the town judge.
There is exactly one person that satisfies properties 1 and 2.
You are given trust, an array of pairs trust[i] = [a, b] representing 
that the person labelled a trusts the person labelled b.

If the town judge exists and can be identified, return the label of the town judge. 
Otherwise, return -1.

 

Example 1:

Input: N = 2, trust = [[1,2]]
Output: 2
Example 2:

Input: N = 3, trust = [[1,3],[2,3]]
Output: 3
Example 3:

Input: N = 3, trust = [[1,3],[2,3],[3,1]]
Output: -1
Example 4:

Input: N = 3, trust = [[1,2],[2,3]]
Output: -1
Example 5:

Input: N = 4, trust = [[1,3],[1,4],[2,3],[2,4],[4,3]]
Output: 3
 
'''

# WHEN GIVEN CONSTRAINTS TO A PROBLEM
# NEGATE THE CONTRAINTS TO EXPLOIT PROBLEM STRUCTURE. think combinatorically 
# about how to use constraints, whether that means to do there exists, or there 
# doesnt exist. especially for greedy questions. 
# think in positive space and negative space.

class Solution(object):
    def findJudge(self, N, trust):
        """
        :type N: int
        :type trust: List[List[int]]
        :rtype: int
        """
        is_judge = [True] * N
        for a, b in trust:
            is_judge[a - 1] = False
        judge_idx = None
        for i, judge in enumerate(is_judge):
            if judge:
                if judge_idx:
                    return -1
                judge_idx = i
        if judge_idx == None:
            return -1
        judge_trust_cnt = 0
        for a, b in trust:
            if b == (judge_idx + 1):
                judge_trust_cnt += 1
        if judge_trust_cnt != N - 1:
            return -1
        return judge_idx + 1
        

class Solution(object):
    def findJudge(self, N, trust):
        """
        :type N: int
        :type trust: List[List[int]]
        :rtype: int
        
        res = [set() for i in range(N) ]
        judge = [i for i in range(1,N+1)]               
     
        for i in trust : 
            res[i[1]-1].add(i[0])
            if i[0] in judge : 
                judge.remove(i[0])
                 
        for j in judge : 
            if len(res[j-1]) == (N-1) :
                return j
               
        return -1 
            
        """
   
        if N == 1:
            return 1
        
        in_degrees = [0 for _ in range(N+1)]
        out_degrees = [0 for _ in range(N+1)]
        for a, b in trust:
            in_degrees[b] += 1
            out_degrees[a] += 1
        
        for i in range(1, N+1):
            if in_degrees[i] == N-1 and out_degrees[i] == 0:
                return i
        
        return -1



from collections import defaultdict

class Solution(object):
    def findJudge(self, N, trust):
        """
        :type N: int
        :type trust: List[List[int]]
        :rtype: int
        """
        '''
        Ok 
        so you just sum it UP!
        
        sum up who everyone likes. 
        or create graph, reverse graph, find node with most trusts,
        and check that the node doesnt point to anything.
        
        '''
        
        if N == 1 and trust == []:
            return 1
        
        
        Xtrusted_by = defaultdict(list)
        Xtrust = defaultdict(list)
        
        for i in trust:
            Xtrusted_by[i[1]].append(i[0])
            Xtrust[i[0]].append(i[1])
        
        
        for k, v in Xtrusted_by.items():
            print("k", k)
            if(len(v) == N-1):
                # Trusted by N-1 ppl, 
                # now we have to check if they trust anyone. 
                if(len(Xtrust[k]) == 0):
                    return k
        
        return -1
        
            
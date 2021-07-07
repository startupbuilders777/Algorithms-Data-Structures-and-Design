'''
TopCoder problem "InformFriends" used in SRM 388 (Division I Level Two)

Problem Statement
You wish to share as many facts as possible with a group of N people,
but you only have time to tell one fact to each person in the group. 
When you tell someone a fact, you also instruct them to tell all their friends. 

However, the friends do not tell their friends: if A and B are friends, 
B and C are friends, but A and C are not friends, then after telling the fact to 
A it will be passed on to B but not to C. You must tell each fact to enough 
people so that every person either hears the fact from you, 
or is a friend of someone who heard it from you.


friends contains N strings of N characters, each of which is either 'Y' or 'N'. 
The jth character of the ith element is 'Y' if members i and j are friends, 
and 'N' otherwise. Determine the maximum number of facts 
that can be shared with every person in the group.

 
Definition
    	
Class:	InformFriends
Method:	maximumGroups
Parameters:	String[]
Returns:	int
Method signature:	int maximumGroups(String[] friends)
(be sure your method is public)
    
 
Constraints
-	friends will contain exactly N elements, where N is between 1 and 15, inclusive.
-	Each element of friends will contain exactly N characters.
-	Each character in friends will be either 'Y' or 'N'.

-	For i and j between 0 and N - 1, inclusive, character j of element i of friends will 
    be the same as character i of element j.

-	For i between 0 and N - 1, inclusive, character i of element i of friends will be 'N'.
 
Examples
0)	
    	
{"NYYN",
 "YNYY",
 "YYNY",
 "NYYN"}
Returns: 3
Tell one fact to people 0 and 3, one fact to 1, and one fact to 2.
1)	
    	
{"NYYN",
 "YNYN",
 "YYNN",
 "NNNN"}
Returns: 1
Person 3 has no friends, and so can learn only one fact directly from you.
2)	
    	
{"NYNNNY",
 "YNYNNN",
 "NYNYNN",
 "NNYNYN",
 "NNNYNY",
 "YNNNYN"}
Returns: 3
Provide facts A, B, C, A, B, C to the six people in order. Each will receive one fact directly and one from each neighbor.
3)	
    	
{"NYNY",
 "YNYN",
 "NYNY",
 "YNYN"}
Returns: 2


'''
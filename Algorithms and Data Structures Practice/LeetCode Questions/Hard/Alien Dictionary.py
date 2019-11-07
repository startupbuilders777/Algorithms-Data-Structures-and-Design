'''
Given a sorted dictionary of an alien language, find order of characters
Given a sorted dictionary (array of words) of an alien language, find order of characters in the language.
Examples:

Input:  words[] = {"baa", "abcd", "abca", "cab", "cad"}
Output: Order of characters is 'b', 'd', 'a', 'c'
Note that words are sorted and in the given language "baa" 
comes before "abcd", therefore 'b' is before 'a' in output.
Similarly we can find other orders.

Input:  words[] = {"caa", "aaa", "aab"}
Output: Order of characters is 'c', 'a', 'b'

'''

# Ok so basically, we keep updating a graph relationship between parent and child!
# I think we can build a graph then just walk the graph to figure shit out right!
# As we look at words, we place characters in graph, and store a parents array!

# Then we do a topological sort right!


# Each word diffs from the next word in exactly one character location. 
# Use this to start building the map
# and iterating through the words. 

'''

ALGORTHM:


1) Create a graph g with number of vertices equal to the size of alphabet in the given alien language. 
    For example, if the alphabet size is 5, then there can be 5 characters in words. 
    Initially there are no edges in graph.

2) Do following for every pair of adjacent words in given sorted array.
    a) Let the current pair of words be word1 and word2. One by one compare characters of both words and 
          find the first mismatching characters.
    b) Create an edge in g from mismatching character of word1 to that of word2.

3) Print topological sorting of the above created graph.


'''

from collections import *
def alien_dictionary(words):
    i = 0
    g = defaultdict(list)
    
    while i+1 < len(words):
        curr_word = words[i]
        next_word = words[i+1]
        
        diff_position = 0 
        smaller_len_word  = min(len(curr_word), len(next_word))
        for idx in range(smaller_len_word):
            if curr_word[idx] != next_word[idx]:
                diff_position = idx
                print(diff_position)
                break
        
        g[curr_word[diff_position]].append(next_word[diff_position])
        i += 1
    
    print(g)
    # TOPO Sort
    from collections import deque
    result = deque()
    visited = set()


    

    def topo_sort(node, visited):

        visited.add(node)
        children = g[node]
        
        for c in children:
            if(c in visited):
                continue

            topo_sort(c, visited)
        result.appendleft(node)
    
    for node in g.keys():
        if node not in visited:
            topo_sort(node, visited)
    
    return result


print(alien_dictionary(["baa", "abcd", "abca", "cab", "cad"]))

print(alien_dictionary(["wrt","wrf","er","ett","rftt"]))

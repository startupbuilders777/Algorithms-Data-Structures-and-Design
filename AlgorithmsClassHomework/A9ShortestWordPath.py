'''
This assignment has three parts: a written part (Q1), a programming part (Q2), and an
optional written bonus part (Q3). You need to hand them in separately as stated above.
The goal is to design and implement (that is, write a working program for) a shortest
path algorithm to solve a problem about word chains.
Please read these instructions carefully before posting any questions to Piazza. We hope
all clarication questions you have will already be answered here. If not, please read previous
questions on Piazza before posting new ones.
We will say two words (strings) x and y are linked from x to y if
    you can get y from x by deleting exactly one letter from x (a \deletion"); or
    you can get y from x by inserting exactly one new letter (an \insertion"); or
    you can get y from x by interchanging two adjacent letters (this is called a twiddle); or
    you can get y by reversing the order of the letters in x ( a \reversal").
For example
 => loose is linked to lose because you can delete one of the o's in loose to get lose;
 => cat is linked to cart because you can insert the letter r;
 => alter is linked to later because you can get one from the other by a twiddle at positions 1 and 2;
 => drawer is linked to reward because you can get one from the other by a reversal.
A word chain is a sequence of words w1;w2; : : : ;wn, where for each i with 1 <= i < n we
have that wi is linked to wi+1. We say the chain links w1 with wn. For example,
spam; maps; map; amp; ramp
is a word chain because

    spam and maps are reversals;
    maps goes to map by a deletion;
    map goes to amp by a twiddle;
    amp goes to ramp by an insertion.

The cost of a word chain is obtained by summing the cost of each link of the chain,
where
    a deletion costs 3;
    an insertion costs 1;
    a twiddle costs 2;
    a reversal of a n-letter word costs n.

If none of these cases apply, you can say that the link has infinite cost.
Thus, for example, the individual costs of the chain above are given as follows:

spam -> 4 -> maps -> 3 map -> 2 -> amp -> 1 -> ramp

and the total cost of the chain is 4 + 3 + 2 + 1 = 10.

In the (very rare) case where two words satisfy two distinct criteria, like tort and trot,
the cost of the link is the lower of the two costs. Here it would be 2.
Given a word list of valid words, and two words u; v from the word list, the word chain
problem is to nd the lowest-cost chain linking u and v using only words from the word list.
The Assignment
The goal is to create a working program to solve the word chain problem in one of C++,
Java, or Python.
'''

# ANSWER TO QUESTION 1 #########################################################
##################################################################################


'''
Q1. [10 marks] This is the written part. Hand it in via Crowdmark.
(a) Discuss how to efficiently solve the word chain problem using a graph algorithm. What
do the vertices represent? What do the edges represent? How are they weighted? In what
form will you store the graph? Is it directed or undirected? Do you expect the graph to be
sparse or dense for a real word list, such as an English dictionary? Do you expect it to be
connected? And, most importantly, how can you efficiently create the graph from a given
word list?
Hint: if you are careful and think about the problem, you can probably create the graph
from a word list rather efficiently. Probably the obvious idea of comparing each pair of words
in the word list, in order to determine the weight of the edge connecting them, is not the best
approach. It should be possible to create the graph in o(n^2) steps (so faster than O(N^2) ), where n is the number of
words, but you might have to make some assumptions about what the word list looks like.
For this part, you are welcome to use techniques like tries, hashing, dictionaries, and so
forth. Make any reasonable assumptions about the running time of anything you use, and
state them. For example, it's reasonable to assume that hashing a string of t letters costs
O(t) time.
We are looking for high-level description here. There is no need to produce pages and
pages of pseudocode.
'''

## PART A ANSWER:

'''

The vertices in the graph will represent each individual word. An edge between 2 vertices will
indicate that the word can be transformed into another word using one of the 4 transformations.
The weight of the edge will be the cost of the transformation (if the transformation can be done in 
multiple ways, take the lowest cost of the 4 transformations). In other words the edges are
weighted by the cost of the transformation. The graph will be undirected because the 4 rules 
that can be followed to transform one word into another have inverses. 
The deletion rule has the addition rule as the inverse and vice versa.
The reverse rule has the reverse rules as its inverse transformation. 
The twiddle rule can be inversed by repeating the twiddle on the same 2 letters.
In other words, if a vertex is connected to another vertex, then they can transform into each other 
using a rule or the inverse rule (all 4 have inverse rules), so the edges will be undirected in the graph.

//CHECK OVER THIS PART:

For a real word list such as the english dictionary, the graph will be very dense for small words 
such as "cat", because there are lots of 3 letter words that can be derived out of cat. For much longer words, 
the graph will be sparse because they will be difficult to transform into other words since longer words have more 
characters that increase the probability of difference between it and another word.  

The graph will be connected because it is undirected.


To efficiently create the graph from a word list, 


we need to find all the words that can be transformed to another word using the 4 rules. 
Comparing 2 strings can be done in O(1). This can be done by doing a 



'''



'''
(b) Next, explain how you can efficiently search the graph to find the solution to the word
chain problem. Here you can assume the graph has been created, and you are given two
words and want to find the lowest-weight chain linking them, or report that there is no such
chain.

What graph algorithm is appropriate? These choices are for you to decide and explain.
Here you should use one of the shortest-path algorithms we've covered.

There's no need to provide pseudocode (unless you want to). Explain your ideas in
English. Provide examples (if they are useful). Be sure to give as much detail as necessary
to convince a skeptical TA.

ANSWER:

Djikstra's algorithm is most suited and most performant for the problem. Djikstra's finds 
the shortest path from one vertex to all others, in O(|E| + |V|log|V|) time. BFS would not
work efficiently because the paths have weights and you would have to break up paths longer 
than 1 unit into connected 1 unit paths. Floyd Warshall is too powerful, and would find the 
shortest paths between all vertices but we do not need to do that for this question. For this
question we only need to find the shortest path from one vertex to another. If the path had 
negative weights, we would have to use the less performant Bellman Ford single source shortest 
path algorithm but since the edges between words are all positive, we can use Dijkstra's algorithm.




'''

'''
(c) Finally, discuss the running time of both (i) creating the graph from the word list and
(ii) given two words, finding the lowest-cost chain connecting them (if it exists).
When you do your analysis, be sure to specify if you're obtaining a rigorous worst-case
bound, or just a \heuristic" bound based on what you think happens in a typical example.
You can express your running time in terms of the appropriate parameters of the word list,
which might include n, the total number of words in the word list, and q, the total number
of symbols of all words in the word list, and m, the maximum length of a word in the word
list.
We will be generous in marking this problem.


Runtime of creating the graph is: 



Runtime of of finding the lowest-cost chain connecting 2 words is:
O(|E| + |v|log|V|) which is the worst case bound for Djikstra's algorithm using a min fibonnaci heap.



UNPLAGARISE THIS:

The complexity of Dijkstra's shortest path algorithm is:

    O(|E| |decrease-key(Q)| + |V| |extract-min(Q)|)
where Q is the min-priority queue ordering vertices by their current distance estimate.

For both a Fibonacci heap and a binary heap, the complexity of the extract-min operation on this queue is O(log |V|). 
This explains the common |V| log |V| part in the sum. For a queue implemented with an unsorted array, the extract-min 
operation would have a complexity of O(|V|) (the whole queue has to be traversed) and this part of the sum would be O(|V|^2).

In the remaining part of the sum (the one with the edge factor |E|), the O(1) v.s. O(log |V|) difference comes precisely 
from using respectively a Fibonacci heap as opposed to a binary heap. The decrease key operation which may happen 
for every edge has exactly this complexity. So the remaining part of the sum eventually has complexity O(|E|) 
for a Fibonacci heap and O(|E| log |V|) for a binary heap. For a queue implemented with an unsorted array, the
decrease-key operation would have a constant-time complexity (the queue directly stores the keys indexed by the vertices) 
and this part of the sum would thus be O(|E|), which is also O(|V|^2).

To summarize:

Fibonacci heap: O(|E| + |V| log |V|)
binary heap: O((|E| + |V|) log |V|)
unsorted array: O(|V|^2)
Since, in the general case |E| = O(|V|^2), these can't be simplified further without making further assumptions on the kind of graphs dealt with.
(https://stackoverflow.com/questions/21065855/the-big-o-on-the-dijkstra-fibonacci-heap-solution)
'''





'''
PROGRAMMING:


Create a program to solve the word chain problem. Your program should have two
distinct parts: a part that reads in a given word list and creates the graph from the word
list, and a part that processes inputs (pairs of words) and determines the chain of lowest
cost connecting the pairs.
Your program should read its word list from a le called dict.txt. Each line of dict.txt
contains a nonempty lower-case word with no symbols other than the lowercase letters from
a to z. Do not assume anything about the ordering of the word list. In particular, it need not
be in alphabetical order. The word list contains words of many dierent (positive) lengths.
3
Once your program has read in the word list, you should create the graph from it. For
testing you can download a dict.txt from the course web page.
For the input of pairs of words, your program should read from standard input and write
to standard output. Your program should expect, as input, the following: t  1 lines of text,
each containing two nonempty words u; v, with the two words on each line separated by a
single space.
There is no guarantee that the input words are actually in the word list. If either one is
not, #print the number 1 for this pair.
(Don't bother checking for inputs that don't adhere to the specs.)
For example, an input might be
spam ramp
tippy sappy
You are to nd the cost of the lowest-cost chain linking u to v, for each of the t pairs
(u; v) of words given, and to give a chain with this cost. (There might be multiple chains of
the given cost; you only have to nd one.)
The output should be the following for each pair of words in the input: the cost of the
chain, followed by a space, followed by the words of the chain from first to last, separated
by one space. If there is no chain, the output should just be the number 1.
For example, for the input above, if dict.txt were
spam
maps
map
amp
sap
sappy
tip
tippy
ramp
the output would be
10 spam maps map amp ramp
-1
Each line of the input and each line of the output has (of course) a newline nn at the
end.
4


'''

# read in word list from dict.txt


def shortestPathWordList():
    from collections import defaultdict
    import heapq
    from sys import stdin
    
    '''
    Add words to adjacency list using a python map
    that maps a key, which is a graph vertex, to an array of values that represent the neighbours
    '''
    graph = defaultdict(list)

    file = open("dict.txt", "r")
    
    # Sort words by their length.
    word_by_lengths = defaultdict(list)


    for line in file:
        word = line.strip()
        word_by_lengths[len(word)].append(word)

    # Only have to compare words with length x with words of length x-1
    # Deletions for words with length x is addition for words of length x-1
    #print(word_by_lengths)
    # get ordered keys from dictionary
    
    # given the graph is dense because it is a dictionary of english words, 
    # there are few unique word lengths so sorting them will be fast and not
    # effect runtime
 
    ordered_word_lengths = sorted(word_by_lengths.keys(), reverse=True)
    #print("the lengths are: ", ordered_word_lengths)


    # for length, words in file: 
    i = 0
    num_of_word_lens = len(ordered_word_lengths)
    shorterWordSet = set()
    longerWordSet = set()

    while i != num_of_word_lens:
        long_word_len = ordered_word_lengths[i]

        longerWords = word_by_lengths.get(long_word_len)

        # check if words of length - 1 exist. If it doesnt, 
        # returns empty array:
        shorterWords = word_by_lengths.get(long_word_len-1)

        # TODO OPTIMIZE THIS, REUSE PREVIOUS LONGWORDSET
        if(shorterWords is None):
            shorterWords = []
        
        # only have to compare these words

        shorterWordSet = set(shorterWords)

        reverseRelationshipChecked = {} 
        twiddleRelationshipChecked = {}

        # check for additions and deletions
        for word in longerWords:
            longerWordSet.add(word) 
            #do every possible deletion, check if it is in the set.
            for position in range(long_word_len):
                possibleShortWord = word[0:position] + word[position + 1: long_word_len]    
                #print("word", word)
                #print("possible word", possibleShortWord)
                

                if(possibleShortWord in shorterWords):
                    # THERE IS AN UNDIRECTED EDGE BETWEEN THESE TWO WORDS!
                    graph[word].append( (possibleShortWord, 3) ) # deletions cost 3. you can actually have a second key which is 
                    graph[possibleShortWord].append( (word, 1)) # insertion costs 1
            
            # check for twiddles and reversals
            
            # dont add relationships twice. Check first.
            
            if(reverseRelationshipChecked.get(word) is None):
                reversed_word = word[::-1]
                if(reversed_word != word and reversed_word in longerWordSet):
                    graph[word].append((reversed_word, long_word_len))
                    graph[reversed_word].append((word, long_word_len))
                reverseRelationshipChecked[reversed_word] = True
            
            
            for position in range(long_word_len - 1):
                twiddleWord = word[:position] + word[position+1] + word[position] + word[position+2:]
                #temp = twiddleWord[position+1]  
                #twiddleWord[position+1] = twiddleWord[position]
                #twiddleWord[position] = temp
                ##print("word is", word)
                ##print("twiddle word is", twiddleWord)

                if(twiddleRelationshipChecked.get(twiddleWord) is None):
                    if(twiddleWord != word and twiddleWord in longerWordSet):
                        graph[word].append((twiddleWord, 2)) #twiddle costs 2
                        graph[twiddleWord].append((word, 2))
                    twiddleRelationshipChecked[twiddleWord] = True
            
                    
        i += 1
        #shorterWordsSet = longerWordSet #Todo: optimizatioon
        shorterWordSet.clear()
        longerWordSet.clear()
    
    def djikistra(graph, wordA, wordB):
        found = False
        visited = set()
        dist = {}
        prev = {}

        dist[wordA] = 0
        
        queue = [(0, wordA)]

        while queue:
            weight, min_word = heapq.heappop(queue)
            
            if(min_word not in visited): 
                visited.add(min_word)
                #print("min word is", min_word)
                if(min_word == wordB): #Found the word
                    found = True
                    break

                for n in graph[min_word]:
                    #print("stfsd", graph[min_word])
                    nb, w = n
                    #print("fck", nb, w)
                    weight = dist[min_word] + w
                    if(weight < dist.get(nb, float("inf"))):
                        dist[nb] = weight
                        prev[nb] = min_word
                        heapq.heappush(queue, (weight, nb))
                        

                #print("THE QUEUE", queue)

        #print("THE PREV", prev)
        #print("THE DIST", dist)
        
        #return dist[wordB]
        if(not found):
            print("-1")
        else:
            printer = str(dist[wordB])
            node = wordB
            lst = []
            while(node != wordA):
                lst.append(prev[node])
                node = prev[node]
            
            for i in reversed(lst) :
                printer += " " + i
            
            printer += " " + wordB
            
            print(printer)
        


    #print(djikistra(graph, "stone", "atone"))
    

    while(True):
        line = raw_input()
        #print(line)
        theIn = line.strip()
        words = theIn.split()
        djikistra(graph, words[0], words[1])
     

shortestPathWordList()
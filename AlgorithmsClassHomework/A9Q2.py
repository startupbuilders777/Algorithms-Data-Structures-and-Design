import heapq
from sys import stdin

def build_graph(graph, edge_weight):
    file = open("flipdict.txt", "r")
    
    for line in file:
        word = line.strip()
        graph[word] = []
        edge_weight[word] = {}
    
    # These maps are used to optimize the checking.
    # If we have seen the word's inverse before, dont add it again.
    reverseRelationshipChecked = {}  
    twiddleRelationshipChecked = {}
    
    for word in graph.keys():
            the_word_len = len(word)

            # longerWordSet.add(word) 
            # do every possible deletion, check if it is in the set.
            for position in range(the_word_len):
                possibleShortWord = word[0:position] + word[position + 1: the_word_len]    
                #print("word", word)
                #print("possible word", possibleShortWord)
                

                if(possibleShortWord in graph):
                    # THERE IS AN UNDIRECTED EDGE BETWEEN THESE TWO WORDS!
                    graph[word].append(possibleShortWord) # deletions cost 3. you can actually have a second key which is
                    edge_weight[word][possibleShortWord] = 3

                    graph[possibleShortWord].append(word) # insertion costs 1
                    edge_weight[possibleShortWord][word] = 1

            # check for twiddles and reversals
            # dont add relationships twice. Check first.
            
            if(reverseRelationshipChecked.get(word) is None):
                reversed_word = word[::-1]
                if(reversed_word != word and reversed_word in graph):
                    #If we have never found this twiddle word before, add it
                    if(edge_weight[word].get(reversed_word) is None): 
                        graph[word].append(reversed_word)
                        edge_weight[word][reversed_word] = the_word_len
                        graph[reversed_word].append(word)
                        edge_weight[reversed_word][word] = the_word_len
                    else:
                        #Otherwise, this is the rare case where we added the reversed word.
                        #Change the weight to the lower of the two
                        edge_weight[word][twiddleWord] = min(edge_weight[word][reversed_word], 2)
                        edge_weight[twiddleWord][word] = min(edge_weight[reversed_word][word], 2)
                
                reverseRelationshipChecked[reversed_word] = True
            
            
            for position in range(the_word_len - 1):
                twiddleWord = word[:position] + word[position+1] + word[position] + word[position+2:]

                if(twiddleRelationshipChecked.get(twiddleWord) is None):
                    if(twiddleWord != word and twiddleWord in graph):
                         #If we have never found this twiddle word before, add it
                        if(edge_weight[word].get(twiddleWord) is None): 
                            graph[word].append(twiddleWord) #twiddle costs 2
                            graph[twiddleWord].append(word)
                            edge_weight[word][twiddleWord] = 2
                            edge_weight[twiddleWord][word] = 2
                        else: 
                            #Otherwise, this is the rare case where we added the reversed word.
                            #Change the weight to the lower of the two
                            edge_weight[word][twiddleWord] = min(edge_weight[word][twiddleWord], 2)
                            edge_weight[twiddleWord][word] = min(edge_weight[word][twiddleWord], 2)

                    twiddleRelationshipChecked[twiddleWord] = True

def shortestPathWordList():    
    '''
    Add words to adjacency list using a python map
    that maps a key, which is a graph vertex, to an array of values that represent the neighbours
    '''
    graph = {}
    edge_weight = {}

    build_graph(graph, edge_weight)
    
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

                for nb in graph[min_word]:
                    #print("stfsd", graph[min_word])
                    w = edge_weight[min_word][nb]

                    #print("fck", nb, w)
                    weight = dist[min_word] + w
                    if(weight < dist.get(nb, float("inf"))):
                        dist[nb] = weight
                        prev[nb] = min_word
                        heapq.heappush(queue, (weight, nb))                 
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
        
    for id, line in enumerate(stdin):
        try:
            theIn = line.strip()
            words = theIn.split()
            djikistra(graph, words[0], words[1])
        except:
            break

shortestPathWordList()

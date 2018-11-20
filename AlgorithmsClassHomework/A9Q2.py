from collections import defaultdict
import heapq
from sys import stdin

def build_graph1(graph):
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
        the_word_len = ordered_word_lengths[i]

        longerWords = word_by_lengths.get(the_word_len)

        # check if words of length - 1 exist. If it doesnt, 
        # returns empty array:
        shorterWords = word_by_lengths.get(the_word_len-1)

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
            for position in range(the_word_len):
                possibleShortWord = word[0:position] + word[position + 1: the_word_len]    
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
                    graph[word].append((reversed_word, the_word_len))
                    graph[reversed_word].append((word, the_word_len))
                reverseRelationshipChecked[reversed_word] = True
            
            
            for position in range(the_word_len - 1):
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

def build_graph2(graph):
    file = open("dict.txt", "r")
    
    for line in file:
        word = line.strip()
        graph[word] = []

        #word_by_lengths[len(word)].append(word)
    
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
                    graph[word].append( (possibleShortWord, 3) ) # deletions cost 3. you can actually have a second key which is 
                    graph[possibleShortWord].append( (word, 1)) # insertion costs 1
            
            # check for twiddles and reversals
            # dont add relationships twice. Check first.
            
            if(reverseRelationshipChecked.get(word) is None):
                reversed_word = word[::-1]
                if(reversed_word != word and reversed_word in graph):
                    graph[word].append((reversed_word, the_word_len))
                    graph[reversed_word].append((word, the_word_len))
                reverseRelationshipChecked[reversed_word] = True
            
            
            for position in range(the_word_len - 1):
                twiddleWord = word[:position] + word[position+1] + word[position] + word[position+2:]
                #temp = twiddleWord[position+1]  
                #twiddleWord[position+1] = twiddleWord[position]
                #twiddleWord[position] = temp
                ##print("word is", word)
                ##print("twiddle word is", twiddleWord)

                if(twiddleRelationshipChecked.get(twiddleWord) is None):
                    if(twiddleWord != word and twiddleWord in graph):
                        graph[word].append((twiddleWord, 2)) #twiddle costs 2
                        graph[twiddleWord].append((word, 2))
                    twiddleRelationshipChecked[twiddleWord] = True
            



def shortestPathWordList():    
    '''
    Add words to adjacency list using a python map
    that maps a key, which is a graph vertex, to an array of values that represent the neighbours
    '''
    graph = defaultdict(list)

    build_graph2(graph)
    
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
    
    '''
    while(True):
        line = raw_input()

        theIn = line.strip()
        words = theIn.split()
        djikistra(graph, words[0], words[1])
     
    '''
    for id, line in enumerate(stdin):
        try:
            theIn = line.strip()
            words = theIn.split()
            djikistra(graph, words[0], words[1])
        except:
            break

shortestPathWordList()
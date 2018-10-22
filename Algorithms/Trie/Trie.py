class TrieNode(object):
    def __init__(self, c):
        self.children = {}
        self.wordEnd = False
        self.isEnd = True

    
        
class Trie(object):
    

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode(None)
        

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        node = self.root
        for i in range(len(word)):
            c = word[i]
            
            if(node.children.get(c) is None):
                node.children[c] = TrieNode(c)
           
            node.isEnd = False
            node = node.children[c]

            
        node.wordEnd = True
    
        # print(node.children)
        

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for i in word:
            # print("in search for i", i, node.children)
            if(node.children.get(i) is None):
                return False
            else:
                node = node.children[i]
        
        return node.wordEnd
    
    
        
    
    def findSuggestions(self, prefix):
        """
        Returns all suggestions that contain the prefix.
        """    
        
        node = self.root
        
        for i in prefix:
            if(node.children.get(i) is None):
                return False
            else:
                node = node.children[i]
        startChar = prefix[-1]
        
        
        def findSuggestions(node, stringSoFar):
            if(node.isEnd):
                return [stringSoFar]
            
            results = []
            if(node.wordEnd):
                results.append(stringSoFar)
                
            for c, i in node.children.items():
                results += findSuggestions(i, stringSoFar + c)
            
            
            return results
        
        
        all_suggestions = findSuggestions(node, startChar)

        
        return all_suggestions
        
    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        
        node = self.root
        for i in prefix:
            if(node.children.get(i) is None):
                return False
            else:
                node = node.children[i]
        
        stack = [node]
        
        while(stack != []):
            
            a = stack.pop()
            
            if(a.wordEnd is True):
                return True
            else:
                for c, i in a.children.items():
                    stack.append(i)
                    
        
        return False

trie = Trie()

trie.insert("hi")
trie.insert("high")

trie.insert("highschool")


print(trie.findSuggestions("h"))

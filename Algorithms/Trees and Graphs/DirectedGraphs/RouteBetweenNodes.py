'''Given a directed graph, design an algorithm to find out whether there is a route between 2 nodes:'''


'''This is a directed graph'''
A = {
    0: [1],
    1: [2],
    2: [0,3],
    3: [2],
    4: [6],
    5: [4],
    6: [5]
}

CS241Example = {
    "a": ["b", "c", "d"],
    "b": ["d"],
    "d": ["a", "c", "f","e"],
    "c": ["a", "e"],
    "e" : ["d"],
    "f" : []
}

'''

DFS ALGO"

'''

def dfsSearch(graph, node1, node2):
    visited = {}
    start = {}
    finish = {}
    timer = [1]

    for i in graph:
        visited[i] = False

    visited[node1] = True
    found = [False]
    dfsSearchRecursive(graph, node1, node2, visited, start, finish, timer, found)
    print("Visited")
    print(visited)
    print("Start")
    print(start)
    print("Finish")
    print(finish)

    return found[0]

def dfsSearchRecursive(graph, node1, node2, visited, start, finish, timer, found):

        start[node1] = timer[0]

        neighbours = graph.get(node1)
        for i in neighbours:
            if(visited.get(i) == False):
                timer[0] += 1
                visited[i] = True
                if(i == node2):
                    found[0] = True
                else:
                    dfsSearchRecursive(graph, i, node2, visited, start, finish, timer, found)
        finish[node1] = timer[0]
        return


class StackNode():
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None

class Stack():
    def __init__(self):
        self.topNode = None

    def push(self, data):
        newNode = StackNode(data)
        if(self.topNode is None):
            self.topNode = newNode
        else:
            self.topNode.next = newNode
            newNode.prev = self.topNode
            self.topNode = newNode

    def top(self):
        if self.topNode is None:
            return None
        else:
            return self.topNode.data

    def pop(self):
        if self.topNode is None:
            return ReferenceError
        else:
            prev = self.topNode.prev
            self.topNode = prev

    def empty(self):
        return self.topNode is None


'''dfs search with a stack and for loop'''




def dfsSearchLoopy(graph, nodeA, nodeB):
    visited = {}
    start = {}
    finish = {}

    for i in graph:
        visited[i] = False

    visited[nodeA] = True
    stack = Stack()
    stack.push(nodeA)

    found = False

    timer = 1

    while not stack.empty():
        node = stack.top()
        stack.pop()

        start[node] = timer
        neighbours = graph[node]
        for i in neighbours:
            if(visited[i] == False):
                timer += 1
                visited[i] = True
                if( i == nodeB ):
                    found = True
                    finish[node] = timer
                    break
                else:
                   # print(i)
                    stack.push(i)
        finish[node] = timer
    print("Visited")
    print(visited)
    print("Start")
    print(start)
    print("Finish")
    print(finish)

    return found


#print(dfsSearch(A, 1, 2))
#print(dfsSearch(A, 0, 2))
#print(dfsSearch(CS241Example, "a", "f"))
print("")
print(dfsSearchLoopy(A, 1, 2))
print(dfsSearchLoopy(A, 0, 2))
print(dfsSearchLoopy(A, 0, 4))
print(dfsSearchLoopy(CS241Example, "a", "f"))
print("")

def bfsSearch(arr, nodeA, nodeB):
    '''Need a queue for this'''
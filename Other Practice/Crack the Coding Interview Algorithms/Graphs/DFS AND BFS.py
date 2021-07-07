class Node:
    def __init__(self, val):
        self.l = None
        self.r = None
        self.v = val

class Tree:
    def __init__(self):
        self.root = None

    def getRoot(self):
        return self.root

    def add(self, val):
        if(self.root == None):
            self.root = Node(val)
        else:
            self._add(val, self.root)

    def _add(self, val, node):
        if(val < node.v):
            if(node.l != None):
                self._add(val, node.l)
            else:
                node.l = Node(val)
        else:
            if(node.r != None):
                self._add(val, node.r)
            else:
                node.r = Node(val)

    def find(self, val):
        if(self.root != None):
            return self._find(val, self.root)
        else:
            return None

    def _find(self, val, node):
        if(val == node.v):
            return node
        elif(val < node.v and node.l != None):
            self._find(val, node.l)
        elif(val > node.v and node.r != None):
            self._find(val, node.r)

    def printTree(self):
        if(self.root != None):
            self._printTree(self.root)

    def _printTree(self, node):
        if(node != None):
            self._printTree(node.l)
            print(str(node.v) + ' ')
            self._printTree(node.r)

    def bfs(self, graph, start):
        visited, queue = set(), [start]
        while stack:
            vertex = queue.pop()
            if vertex not in visited:
                visited.add(vertex)
                # new nodes are added to end of queue
                queue.extend(graph[vertex] - visited)
        return visited

    def dfs(self, graph, start):
        visited, stack = set(), [start]
        while queue:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                # new nodes are added to the start of stack
                stack = graph[vertex] - visited + stack
        return visited
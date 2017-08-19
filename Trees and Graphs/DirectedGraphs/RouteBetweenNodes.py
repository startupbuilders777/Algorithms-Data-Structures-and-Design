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


#print(dfsSearch(A, 1, 2))
#print(dfsSearch(A, 0, 2))
print(dfsSearch(CS241Example, "a", "f"))


'''dfs search with a stack and for loop'''



dfsSearchLoopy(graph, node1, node2):

# NOTE DONE
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        
        #So basically do a dfs, remove courses from the set as you visit them
        # we are basically checking for cycles. 
        # cycles can be in seperate coordinates.
        
        #coursesToTake = set([i in range(0, numCourses)])
        m = {} # map that indicates directed relationships in graph
        
        
        for i in prerequisites:
            if(m.get(i[0]) is None):
                m[i[0]] = [i[1]]
            else:
                m[i[0]].append(i[1])
        
        
        # do a dfs with a stack
        
        all_nodes = set([i for i in range(0, numCourses)])
        # deque is same as list, but has popleft so you can use it as a queue. for stack, 
        # you do pop on deque, which removes right side of deque. both deque and list have append. deque has append left
        # dont use list to pop(0) because that is O(N) which is bad should be O(1)
        
        
        '''
        OTHER COOL SET OPERATIONS:
        s.issubset(t)	s <= t	test whether every element in s is in t
        s.issuperset(t)	s >= t	test whether every element in t is in s
        s.union(t)	s | t	new set with elements from both s and t
        s.intersection(t)	s & t	new set with elements common to s and t
        s.difference(t)	s - t	new set with elements in s but not in t
        s.symmetric_difference(t)	s ^ t	new set with elements in either s or t but not both
        s.copy()	 	new set with a shallow copy of s
        
        s.update(t)	s |= t	return set s with elements added from t
        s.intersection_update(t)	s &= t	return set s keeping only elements also found in t
        s.difference_update(t)	s -= t	return set s after removing elements found in t
        s.symmetric_difference_update(t)	s ^= t	return set s with elements from s or t but not both
        s.add(x)	 	add element x to set s
        s.remove(x)	 	remove x from set s; raises KeyError if not present
        s.discard(x)	 	removes x from set s if present
        s.pop()	 	remove and return an arbitrary element from s; raises KeyError if empty
        s.clear()	 	remove all elements from set s
        
        '''
        nodes_in_different_component = set()
        
        print("m", m)
        while(len(all_nodes) > 0):
            randomElem = all_nodes.pop()
            
            stack, visited = [randomElem], set()
            print("VISITED START", visited)
            
            print("STACK STARTS WITH ", stack)
            while(stack != []):
                elem = stack.pop()
             
                if(elem in visited):
                    print("check elem", elem)
                    print("check visiited in if", visited)
                    return False # we found a cycle through the DFS
                
                visited.add(elem)
                print("curr visited", visited)
                prerequisites = m.get(elem) if m.get(elem) is not None else [] 
                print(prerequisites)
                
                for prerequisite in prerequisites:
                    # you cant revisit nodes that have already been visited in a "different component"
                    if(prerequisite not in nodes_in_different_component):
                        stack.append(prerequisite)
            
            all_nodes.add(randomElem) #add back random element
            print("all_nodes before", all_nodes)
            all_nodes -= visited
            nodes_in_different_component |= visited
            
            print("all nodes after ", all_nodes)
        
        return True
            
        
            
        # remove all visited nodes from all_nodes set, and find cycles in other components.
            
            
        
        
        
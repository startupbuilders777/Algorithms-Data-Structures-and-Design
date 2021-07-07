'''

841. Keys and Rooms



There are N rooms and you start in room 0.  Each room has a distinct number in 0, 1, 2, ..., N-1, 
and each room may have some keys to access the next room. 

Formally, each room i has a list of keys rooms[i], and each key rooms[i][j] is an integer in [0, 1, ..., N-1] 
where N = rooms.length.  A key rooms[i][j] = v opens the room with number v.

Initially, all the rooms start locked (except for room 0). 

You can walk back and forth between rooms freely.

Return true if and only if you can enter every room.

Example 1:

Input: [[1],[2],[3],[]]
Output: true
Explanation:  
We start in room 0, and pick up key 1.
We then go to room 1, and pick up key 2.
We then go to room 2, and pick up key 3.
We then go to room 3.  Since we were able to go to every room, we return true.
Example 2:

Input: [[1,3],[3,0,1],[2],[0]]
Output: false
Explanation: We can't enter the room with number 2.
Note:

1 <= rooms.length <= 1000
0 <= rooms[i].length <= 1000
The number of keys in all rooms combined is at most 3000.

'''


class Solution:
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        # run a dfs with a visited set, or a bfs. encode both!
        # basically testing if this directed graph is connected! start from room 0, and keep going.

        visited, roomStack = set(), []
    
        roomStack.append(0)
        
        while roomStack != []:
            room = roomStack.pop() 
            if(room in visited):
                continue 
            else:
                visited.add(room)
                
            for i in rooms[room]:
                # i is the key
                roomStack.append(i)
            
        
        if len(visited) == len(rooms):
            return True
        else:
            return False

    
# official solution

class Solution(object):
    def canVisitAllRooms(self, rooms):
        seen = [False] * len(rooms)
        seen[0] = True
        stack = [0]
        #At the beginning, we have a todo list "stack" of keys to use.
        #'seen' represents at some point we have entered this room.
        while stack:  #While we have keys...
            node = stack.pop() # get the next key 'node'
            for nei in rooms[node]: # For every key in room # 'node'...
                if not seen[nei]: # ... that hasn't been used yet
                    seen[nei] = True # mark that we've entered the room
                    stack.append(nei) # add the key to the todo list
        return all(seen) # Return true iff we've visited every room



# FASTEST SOLUTION

class Solution:
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        seen = [False] * len(rooms)
        seen[0]  = True;
        stack = [];
        stack.append(0);
        
        while (stack):
            index = stack.pop();
            for i in rooms[index]:
                if not seen[i]:
                    seen[i] = True;
                    stack.append(i)
        return all(seen)
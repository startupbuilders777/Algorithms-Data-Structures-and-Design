# Python3 program to find Maximum 
# number of partitions such that 
# we can get a sorted array. 
  
# Function to find maximum partitions. 
def maxPartitions(arr): 
  
    smallest_elements = [float("inf")]

    for i in arr:
        print("process", i)
        while smallest_elements and i < smallest_elements[-1]:
            print(i)
            smallest_elements.pop()
        smallest_elements.append(i)
        print("smallest elements", smallest_elements)


    return len(smallest_elements)



# Driver code 
arr = [-3, -2,0,1,2,3]  
n = len(arr) 
print(maxPartitions(arr)) 
  
# This code is contributed by Smitha Dinesh Semwal. 


def heapify(A):
    for root in xrange(len(A)//2-1, -1, -1):
        rootVal = A[root]
        child = 2*root+1
        while child < len(A):
            if child+1 < len(A) and A[child] > A[child+1]:
                child += 1
            if rootVal <= A[child]:
                break
            A[child], A[(child-1)//2] = A[(child-1)//2], A[child]
            child = child *2 + 1

arr = [1, 4, 1, 3, 6 , 8 , 9, 1 , 3 , 56, 5]
heapify(arr)

print(arr)


#include <stdio.h>
#include <limits.h>
#include <iostream> 


/*
Question about Q1 HW4
Hello!
(HW4,Q1)
 
I am confusing myself with the "non-empty" condition even though I feel I might be correct.
 
[-3, -4, -9]
For this case, is the answer 0 or is it -3?
 
I am tricking myself because I believe it should be 0, because I can have [-3] as the single subarray and then remove the -3. 
 
But the non-emptiness is confusing me. If I have [-3] and I remove it, I have an empty subarray so this fails the question condition.
 
Which interpretation is correct?



The answer should be 0, for the following reasoning:
First, you get a "nonempty" subrange with only one element [-3].
Then, when you calculate the "omit-one sum", you get value 0 because the only element in the subrange will be omitted.
 
*/

// Utility function to find maximum of two numbers
int max(int x, int y) {
    return (x > y) ? x : y;
}

int min(int x, int y) {
	return (x < y) ? x : y;
}

// Function to find maximum subarray sum using divide and conquer
int maximum_sum(int A[], int low, int high)
{
    // If array contains only one element
    if (high == low)
        return A[low];

    // Find middle element of the array
    int mid = (low + high) / 2;

    // Find maximum subarray sum for the left subarray
    // including the middle element
    int left_max = INT_MIN;
    int left_max_without_min = INT_MIN;
    int sum = 0;
    int curr_min_left = INT_MAX;
    
    for (int i = mid; i >= low; i--)
    {
        sum += A[i];
        
        if(A[i] < curr_min_left) {
            curr_min_left = A[i];
        }

        left_max = max(left_max, sum);
        left_max_without_min = max(left_max_without_min, sum-curr_min_left);
    }

    // Find maximum subarray sum for the right subarray
    // excluding the middle element
    int right_max = INT_MIN;
    int right_max_without_min = INT_MIN;
    sum = 0;
    int curr_min_right = INT_MAX;
    
    for (int i = mid + 1; i <= high; i++)
    {
        sum += A[i];

        if(A[i] < curr_min_right) {
            curr_min_right = A[i];
        }
        
        right_max = max(right_max, sum);
        right_max_without_min = max(right_max_without_min, sum-curr_min_right);
    }

    // we have to choose a a side that has the min pulled and 
    // a side that doesn't have a min pulled

    int boundary_max = max(left_max_without_min + right_max, left_max + right_max_without_min); 


    // Recursively find the maximum subarray sum for left subarray
    // and right subarray and take maximum
    int max_left_right = max(maximum_sum(A, low, mid),
                            maximum_sum(A, mid + 1, high));

    // return maximum of the three
    //std::cout << "min_left: " << minleft << " min_right: " << minright << std::endl;
    //std::cout << "max_left: " << left_max << " max_right: " << right_max << std::endl;
    return max(max_left_right, boundary_max); 
    // are we removing the min multiple times. No
}

// Maximum Sum Subarray using Divide & Conquer
int main(void)
{
    int arr[] = {5, -4, 1, 2, 3, -4, 5, -20000};
    int arr2[] = {10, -200000, 1,-3, 2, -10000, 11}; 
    
    int n = sizeof(arr) / sizeof(arr[0]);   
    int n2 = sizeof(arr2) / sizeof(arr2[0]);

    printf("The maximum sum of the subarray is %d", maximum_sum(arr2, 0, n2 - 1));

    return 0;
}

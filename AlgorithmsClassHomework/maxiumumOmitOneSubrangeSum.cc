#include <stdio.h>
#include <limits.h>
#include <iostream> 

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
    int sum = 0;
    int minleft = INT_MAX;
    for (int i = mid; i >= low; i--)
    {
        sum += A[i];
        if (sum > left_max) {
            left_max = sum;
            if (minleft > A[i]) minleft = A[i];
        }
    }

    // Find maximum subarray sum for the right subarray
    // excluding the middle element
    int right_max = INT_MIN;
    sum = 0;
    int minright = INT_MAX;
    for (int i = mid + 1; i <= high; i++)
    {
        sum += A[i];
        if (sum > right_max)
            right_max = sum;
            if (minright > A[i]) minright = A[i];
    }

    // Recursively find the maximum subarray sum for left subarray
    // and right subarray and take maximum
    int max_left_right = max(maximum_sum(A, low, mid),
                            maximum_sum(A, mid + 1, high));

    // return maximum of the three
    std::cout << "min_left: " << minleft << " min_right: " << minright << std::endl;
    std::cout << "max_left: " << left_max << " max_right: " << right_max << std::endl;
    return max(max_left_right, left_max + right_max - min(minright, minleft));
}

// Maximum Sum Subarray using Divide & Conquer
int main(void)
{
    int arr[] = {5, -4, 1, 2, 3, -4, 5, -20000};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("The maximum sum of the subarray is %d", maximum_sum(arr, 0, n - 1));

    return 0;
}
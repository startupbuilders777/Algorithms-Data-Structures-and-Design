'''
Given an array of integers, find if the array contains any duplicates. Your function should return true if any value 
appears at least twice in the array, and it should return false if every element is distinct.

'''

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        setX = {}
        for i in nums:
            if(setX.get(i) is None):
                setX[i] = 1
            else:
                return True
        return False

'''
FASTER SOLUTIONS
'''
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        duplicates = set()
        for num in nums:
            if num not in duplicates:
                duplicates.add(num)
            else:
                return True
        return False

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        return len(nums) != len(set(nums))
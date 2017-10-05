'''
Given two arrays, write a function to compute their intersection.

Example:
Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].

Note:
Each element in the result must be unique.
The result can be in any order.

'''


class Solution(object):
    # This solution is if you want to return duplicates so if
    # You have nums1 = [1, 2, 2, 1], nums2 = [2, 2], it returns [2, 2]
    def intersection2(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        intersectionMap = {}
        # Performance is O(n+m)

        for i in nums1:
            if (intersectionMap.get(i) is None):
                intersectionMap[i] = 1
            else:
                intersectionMap[i] += 1

        intersectionLst = []

        for i in nums2:
            if (intersectionMap.get(i) is not None):
                intersectionLst.append(i)
                intersectionMap[i] -= 1
                if (intersectionMap[i] == 0):
                    del intersectionMap[i]

        return intersectionLst

    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        intersectionMap = {}
        # Performance is O(n+m)

        for i in nums1:
            if (intersectionMap.get(i) is None):
                intersectionMap[i] = 1
            else:
                intersectionMap[i] += 1

        intersectionLst = []

        for i in nums2:
            if (intersectionMap.get(i) is not None):
                intersectionLst.append(i)
                del intersectionMap[i]

        return intersectionLst

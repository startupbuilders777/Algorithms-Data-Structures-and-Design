class Solution:
    # if the number has a factor, then check if it has a second factor of the same type.
    # 81 -> 3 is factor, now check if 27 has 3 as a factor -> yes keep going now you have to check if 9 is a power of 2.
    # 100 -> 2 is a factor, check if 50 has a factor of 2 aswell. OK good now run and see that powerOf2 on 25 works. Ok good job.

    def isPerfectSquare(self, n):
        """
        :type n: int
        :rtype: bool
        """

        def checkDivisibleBy(number, divisor):
            return number % divisor == 0

        if (n == 1):
            return True

        counter = 2;
        while True:
            if (counter == n):
                return False
            if (checkDivisibleBy(n, counter)):
                n /= counter
                if (checkDivisibleBy(n, counter)):
                    n /= counter
                    return self.isPerfectSquare(n)
                else:
                    return False

            counter += 1

soln = Solution()
print(soln.isPerfectSquare(100))
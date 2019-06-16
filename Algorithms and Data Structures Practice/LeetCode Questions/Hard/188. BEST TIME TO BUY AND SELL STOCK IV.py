# NOT DONE 

'''
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most k transactions.

Note:
You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

Example 1:

Input: [2,4,1], k = 2
Output: 2
Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.
Example 2:

Input: [3,2,6,5,0,3], k = 2
Output: 7
Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4.
             Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.

'''

class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        # YOU MAY COMPLETE AT MOST K TRANSACTIONS!
        
        # ok so dp? prolly
        
        '''
        
        
        Start from the back of the array. 
        
        process backwards. 
        
        [2, 4, 1]
        
        you see 1, then see 4. you didnt buy anything so avoid that
        
        now you at 4, this our curr_max
        
        go backwards, find 2. ok go again. 
        (until you find a smaller min) stop 
        when you hit start of array          
        or a value bigger than 4!
        then do the exchange. 
        that is one transaction
        
        we are allowed to at most K.
        
        so we will do K if all K lead to a profit. 
        
        if there are less than K do all of them.
        If there are more than K, we have to pick the best ones!
        
        can keep a max heap of size K. if its bigger than the root, 
        then insert the new element, pop smallest element in max heap?
         
        maybe use a deque!
        no
        i think insertion into an array is good!
        
        
        anyway i will do sort, and just take top k.
        That is nlog(n) fooook
        
        '''
        
        curr_max = float("-inf")
        have_stock = False
        stock_purchase_price = None
        
        profits = []
        surges = []
        for i in range(len(prices)-1, -1,-1):
            print("i", i)
            if(prices[i] > curr_max):
                # sell stock
                if(have_stock):
                    profit = curr_max - stock_purchase_price
                    profits.append(profit)
                    have_stock  = False
                    surges.append((curr_max, stock_purchase_price))
                curr_max = prices[i]
                
            elif(not have_stock):
                have_stock = True
                stock_purchase_price = prices[i]
            elif(have_stock and prices[i] < stock_purchase_price):
                stock_purchase_price = prices[i]
            elif(have_stock):
                # we have a stock!, and we just saw a higher price! sell now!
                # then save that as the new curr_max
                profit = curr_max - stock_purchase_price
                profits.append(profit)
                surges.append((curr_max, stock_purchase_price))
                have_stock  = False
                
                curr_max = prices[i]
                
                
        # keep track of all the buy, sell pieces!
        # then when you pick k, just 
        # merge intervals so that there are k intervals!
        # not sort!
        
        if(have_stock):
            profit = curr_max - stock_purchase_price
            profits.append(profit)
            surges.append((curr_max, stock_purchase_price))
            have_stock  = False
        
        sorted_profits = sorted(profits, reverse=True)
        # print("surges", surges)
        # FOR THE SURGES!, MERGE ADJACENT SURGES THAT HAVE THE LEAST PROFIT. 
        # REPEAT PROCESS UNTIL THERE ARE K INTERVALS!
        def get_left_not_none(arr, i):
            counter = i - 1
            while(counter >= 0):
                if(arr[counter] is not None):
                    return (counter, arr[counter])
                counter -= 1
            
            return (None, None)
        
        def get_right_not_none(arr, i):
            counter = i + 1
            while(counter < len(arr)):
                if(arr[counter] is not None):
                    return (counter, arr[counter])
                counter += 1
            return (None, None)
            
            
        def merge_surges_into_k(surges, profits, k):
            import heapq
            import itertools
            '''
            How to maintain a sorted list profits so we can pop the worst one each time!
            use bisect i guess!
            or whatever!
            
            OR use a dictionary!
            still not goood enuff!
            USE A HEAPQ and do change key
            '''
            worst_surges_pq = []
            
            worst_surge_finder = {}
            counter = itertools.count()
            REMOVED = None # placeholder for removed task
            
            for idx, profit in enumerate(profits):
                entry = [profit, next(counter), idx]
                worst_surge_finder[idx] = entry
                heapq.heappush(worst_surges_pq, entry)
            
            # print("worst surges", worst_surges_pq)
            
            surges_len = len(surges)
            amt_to_merge = surges_len - k
            # to merge, take max of sell side for both, and min of buy side for both
            i = 0
            print("surges are", surges)
            print("profts", profits)
            
            print("max proftt possible", sum(profits))
            
            while i < amt_to_merge:
                # find least profitable surges, try merging to left side, and right side. 
                # keep the one that creates more profit!
                # print("AMT_TO_MERGE I", i)
                # idx = worst_surges[i][0] # get index of worst surge 
                pri, count, idx = heapq.heappop(worst_surges_pq)
                if(idx is REMOVED):
                    continue
                
                i += 1
                
                left_merge = (0, 0)
                right_merge = (0, 0)
                
                left_interval_idx, left_interval = get_left_not_none(surges, idx)
                right_interval_idx, right_interval = get_right_not_none(surges, idx)
                
                left_interval_used_it = False
                right_interval_used_it = False
                print("SURGE POOPEED IS", surges[idx])
                
                if(left_interval_idx):
                    # you have two intervals:
                    # left interval can use right interval's min as a possiblity
                    # left interval only keep no merge
                    # right interval we throw away no matter what because its garbage rite since it is min. 
                    #
                    # SO WE ARE COMPARING LEFT INTERVAL, WITH LEFT INTERVAL GETTING AID FROM worst surge's min
                    
                    if(surges[idx][1] < left_interval[1]):
                        left_merge = (left_interval[0], surges[idx][1])
                        left_interval_used_it = True
                    else:
                        left_merge = left_interval
                
                if(right_interval_idx):
                    # two intervals, 
                    # our interval is shit. maybe the right interval can use our sell point 
                    # to make a better profit rite?
                    # COMPARE NORMAL RIGHT INTERVAL WITH RIGHT INTERVAL GETTING AID FROM OUR MIN?
                    if(surges[idx][0] > right_interval[0]):
                        right_merge = (surges[idx][0], right_interval[1])
                        right_interval_used_it = True
                    else: 
                        right_merge = right_interval
                
                best_merge = None
                interval_changed_idx = None
                if(left_interval_used_it and 
                   right_interval_used_it and 
                   left_merge[0] - left_merge[1]  > right_merge[0] - right_merge[1]):
                    # choose left merge
                    # update left 
                    # continue
                    # also change its current profit rite!
                    # this might need a heap tbh! fok
                    best_merge = left_merge
                    surges[left_interval_idx] = left_merge
                    interval_changed_idx = left_interval_idx
                    
                    # I want to pop an interval out at O(1) time. should have used ordered map!
                elif(left_interval_used_it and right_interval_used_it):
                    best_merge = right_merge
                    surges[right_interval_idx] = right_merge
                    interval_changed_idx = right_interval_idx
                elif(left_interval_used_it):
                    best_merge = left_merge
                    surges[left_interval_idx] = left_merge
                    interval_changed_idx = left_interval_idx
                elif(right_interval_used_it):
                    best_merge = right_merge
                    surges[right_interval_idx] = right_merge
                    interval_changed_idx = right_interval_idx
                else:
                    pass # who cares. code below handles that
                    
                surges[idx] = None # or just set it to None i guess!
                    
                    
                # THIS IS HOW YOU DO UPDATE KEY IN A HEAPQ
                # also have to pop the interval out of heapq and then put in new key!
                # remove task!
                if(left_interval_used_it or right_interval_used_it):
                    entry = worst_surge_finder.pop(interval_changed_idx)
                    entry[-1] = REMOVED
                    new_entry = [best_merge[0] - best_merge[1], 
                                                     next(counter), 
                                                     interval_changed_idx]
                    heapq.heappush(worst_surges_pq, new_entry )
                    worst_surge_finder[interval_changed_idx] = new_entry
                    
                    
                print("best merge is", best_merge)

            print("after merging!", surges)
            return surges
        
                    
        k_surges = merge_surges_into_k(surges, profits, k)
        # print("k surges", k_surges)
        surge_profits = map(lambda surge: surge[0] - surge[1] if surge else 0, k_surges)
        print("surge profits", list(surge_profits))
        return sum(map(lambda surge: surge[0] - surge[1] if surge else 0, k_surges))
    
        
        
            
sol = Solution()
print(sol.maxProfit(2, [1,2,4,2,5,7,2,4,9,0,9]))
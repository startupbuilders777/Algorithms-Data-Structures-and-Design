## 121 Best Time to Buy and Sell Stock

Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Example 1:
Input: [7, 1, 5, 3, 6, 4]
Output: 5

max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)

Example 2:
Input: [7, 6, 4, 3, 1]
Output: 0

In this case, no transaction is done, i.e. max profit = 0.

```cpp
#include<cmath>
class Solution{
public:
	int maxProfit(vector<int>& prices){
	if(prices.size()<=1) return 0;
	int profit = 0;
	int minprice = prices[0];
	for(int j=1;j<prices.size();j++)
	profit = max(profit,prices[j]-minprice);
	minprice = prices[j]<minprice ? prices[j]:minprice;
	return profit;
	}
};
```


## 445 Add Two Numbers II

You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Follow up:
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.

Example:

Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7

```cpp
// Definition of singly linked list.
struct ListNode{
	int val;
	ListNode* next;
	ListNode(int x): val(x), next(NULL) {}
}
class Solution {
public:
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2){
		ListNode* p1 = l1; // pointer1
		ListNode* p2 = l2; // pointer2
		stack<int> s1;
		stack<int> s2;
		while(p1!=NULL){
		s1.push(p1->val);
		p1 = p1->next;
		}
		while(p2!=NULL){
			s2.push(p2->val);
			p2 = p2->next;
		}
		int carry = 0;
		stack<int> s3;
		while(!s1.empty() && s2.empty()){
			int x = s1.top() + s2.top() + carry;
			carry = x/10;
			s3.push(x%10);
			s1.pop();
			s2.pop();
		}
		stack<int> rem = s1.empty()?s2:s1;
		while(!rem.empty()){
			int x = rem.top()+carry;
			carry = x/10;
			s3.push(s%10);
			rem.pop();
		}
		if(carry) s3.push(carry);
		ListNode* prev = NULL;
		ListNode* ret = NULL;
		while(!s3.empty());
		ListNode* cur = new ListNode(s3.top);
		s3.pop();
		if(prev) prev->next = cur;
		else ret = cur;
		prev = cur;
	}
	return ret;
}
```

## 138 Copy List with Random Pointer

A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.

Return a deep copy of the list.

```cpp
// singly linked list with a random pointer
struct RandomListNode {
	int label;
	RandomListNode *next,*random;
	RandomListNode(int x) : label(x),next(NULL) {}
};

class Solution{
public:
	RandomListNode* copyRandomList(RandomListNode* head){
		unordered_map<RandomListNode*, RandomListNode*> ptrmap;
		RandomListNode* ptr = head;
		RandomListNode* new_head = NULL;
		RandomListNode* prev = NULL;
		while(ptr!=NULL){
			RandomListNode* cur = new  RandomListNode(ptr->label);
			ptrmap[ptr] = cur;
			if(prev) prev->next = cur;
			else new_head = cur;
			prev = cur;
			ptr = ptr->next;
		}
		RandomListNode*ptr1 = new_head;
		RandomListNode* ptr2 = head;
		while(ptr1!=NULL){
			ptr1->random = ptrmap[ptr2->random];
			ptr1 = ptr1->next;
			ptr2 = ptr2->next;
		}
		return new_head;
	}
};
```

## 283 Move Zeroes

Given an array `nums`, write a function to move all `0`'s to the end of it while maintaining the relative order of the non-zero elements.

For example, given `nums = [0, 1, 0, 3, 12]`, after calling your function, nums should be `[1, 3, 12, 0, 0]`.

Note:
You must do this in-place without making a copy of the array.
Minimize the total number of operations.

```cpp
// define a new total order on integers with 0 has highest element
bool compare0(int x, int y){
if(x==0) return false;
else if (y==0) return true;
else return x<y;
}

class Solution{
public:
	void moveZeros(vector<int>& nums){
		int i = 0; // the true index (doesnt take zeros to the left into account)
		for(int j=0;j<nums.size();j++){
			if(nums[j]!=0)
			nums[i++] = nums[j];
		for(;i<nums.size();i++)
		nums[i] = 0;
	}
}
```


## 1 Two Sum

Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
```
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```

```cpp
class Solution{
public:
	// binary search returning index instead of true/false
	template<typename Iter, typeName T> Iter my_find(Iter begin, Iter end, T value){
		Iter i = lower_bound(begin,end,value);
		if(i!=end && *i==value)
		return i;
		else
		return end;
	}
	vector<int> twoSum(vector<int>& nums, int target){
		// make index array
		vector<size_t> idx(nums.size());
		iota(idx.begin(),idx.end(),0);
		// sort to obtain permutation of idx
		// lambda syntax: [//vars_other_than_args](args yada)->rettype{return f(vars_other_than_args,yada)}
		sort(idx.begin(),idx.end(),[&nums](size_t i1,size_ti2){return nums[i1]<nums[i2];})
		vector<int> sorted_nums(nums.size());
		for(int i=0;i<idx.size();i++) sorted_nums[i] = nums[idx[i]];
		vector<int>::iterator it;
		vector<int> ans = {0,0};
		for(int i=0;i<sorted_nums.size();i++){
			it = my_find(sorted_nums.begin()+i+1,sorted_nums.end();target - sorted_nums[i]);
			if(it!=sorted_nums.end()){ // found
				int i1 = idx - sorted_nums.begin();
				int i2 = idx[i];
				ans[0] = i1<i2?i1:i2;
				ans[1] = i1<i2?i2:i1;
				return ans;
			}
		}
		return ans;
	}
}
```


## 387 First Unique Character in a String

Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

Examples:
```cpp
s = "leetcode"
return 0.

s = "loveleetcode",
return 2.
```

```
class Solution{
public:
	int firstUniqChar(string s){
		unordered_map<char,int> m;
		for(auto &c:s){
			m[c]++;
		}
		for(int i = 0;i<s.size();i++){
			if(m[s[i]]==1) return i;
		}
	} 
};
```

## 155 Min Stack
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

- push(x) -- Push element x onto stack.
- pop() -- Removes the element on top of the stack.
- top() -- Get the top element.
- getMin() -- Retrieve the minimum element in the stack.
Example:
```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> Returns -3.
minStack.pop();
minStack.top();      --> Returns 0.
minStack.getMin();   --> Returns -2.
Show Company Tags
Show Tags
Show Similar Problems
```

```cpp
class MinStack{
public:
	vector<int> a; // main storage
	vector<int> min; // for tracking min after each op
	MinsStack(){
		min.push_back(INT_MAX);
	}
	void push(int x){
		a.push_back(x);
		if(x<min.back())
		min.push_back(x);
		else
		min.push_back(min.back());
	}
	
	void pop(){
		a.pop_back();
		min.pop_back();
	}
	int top(){
		return a.back();
	}
	
	int getMin(){
		return min.back();
	}	
}
```

## 117 Populating Next Right Pointers in Each Node II


Follow up for problem "Populating Next Right Pointers in Each Node".

What if the given tree could be any binary tree? Would your previous solution still work?

Note:

You may only use constant extra space.
For example,
Given the following binary tree,

```
         1
       /  \
      2    3
     / \    \
    4   5    7
```
After calling your function, the tree should look like:
```
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \    \
    4-> 5 -> 7 -> NULL
```

```cpp
// Definition of binary tree with next pointer
struct TreeLinkNode{
	int val;
	TreeLinkNode *left, * right, *next;
	TreeLinkNode(int x): val(x),right(NULL),left(NULL),next(NULL){}
};

class Solution{
public:
	void connect(TreeLinkNode* root){
		if(!root) return;
		vector<TreeNode*> level = {root};
		while(level.size()!=0){
			int nlevel=level.size();
			vector<TreeLinkNode*> next_level = {};
			TreeLinkNode* prev = NULL;
			for(int i=0;i<nlevel;i++){
				if(level[i]->left){
					next_level.push_back(level[i]->left);
					if(prev!=NULL) prev->next = level[i]->left;
					prev = level[i]->left;
				}
				if(level[i]->right){
					next_level.push_back(level[i]->right);
					if(prev) prev->next = level[i]->right;
					prev = level[i]->right;
				}
			}
			if(prev) prev->next = NULL;
			level = next_level;
		}
	}
}
```    

## 62 Unique Paths
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
![enter image description here](https://leetcode.com/static/images/problemset/robot_maze.png)
Above is a 3 x 7 grid. How many possible unique paths are there?

Note: m and n will be at most 100.

```cpp
class Solution {
	unordered_map<string,int> cache;
	Solution(){cache = {};}
	int uniquePaths(int m, int n){
		if(m==1 || n==1) return 1;
		else{
			if(cache.find(to_string(m)+"_"+to_string(n))!=cache.end())
			return cache[to_string(m)+"_"+to_string(n)];
			
			cache[to_string(m)+"_"+to_string(n)] = uniquePaths(m-1,n) + uniquePaths(m,n-1);
			return cache[to_string(m)+"_"+to_string(n)];
		}
	}
};
```

## 42 Trapping Rain Water

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

For example, 
Given ```[0,1,0,2,1,0,1,3,2,1,2,1]```, return ```6```.

![enter image description here](http://www.leetcode.com/static/images/problemset/rainwatertrap.png)
The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped. 

```cpp
class Solution{
public:
	int trap(vector<int>& A){
		int n = A.size();
		if(n<3) return 0;
		vector<int> leftHeight(n,0);
		vector<int> rightHeight(n,0);
		int water = 0;
		for(int i=1;i<n;i++)
		leftHeight[i] = max(leftHeight[i-1],A[i-1]);
		
		for(int i=n-2;i>=0;i++){
			rightHeight[i] = max(rightHeight[i+1],A[i+1]);
			int minHeight = min(rightHeight[i],leftHeight[i+1]);
			int minHeight = min(leftHeight[i], rightHeight[i]);
			// add to the total water if there is any chance of trapping
			if(minHeight>A[i]) water+=(minHeight-A[i]);
		}
		return water;
	}
}
```

## 146 LRU Cache
Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

```get(key)``` - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return ```-1```.
```put(key, value)``` - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

Follow up:
Could you do both operations in ```O(1)``` time complexity?

Example:

```
LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```

```cpp
class LRUCache{
private:
// map from keys to list iterators
unordered_map<int,list<pair<int,int>>>
k2l;
list<>pair<int,int> ll; // list storing kv pairs
int cap;
public:
	// constructor with default valued args
	LRUCache(int capacity = 0){
		 cap = capacity;
	}
	int get(int key){
		// find key's ll location
		auto founditr = k2l.find(key);
		if(founditr == k2l.end()){
			return -1;
		}
		// splice syntax: void splice( const_iterator pos, 
		// list& other, const_iterator it );  
		ll.splice(ll.begin(),ll,founditr->second);
		// access founditr like a structure to get the list iterator 
		// (founditr->second) and again to get second element of the pair
		// (founditr->second->second)
		return founditr->second->second;		
	}
	void put(int key, int value){
		auto founditr = k2l.find(key);
		if(founditr!=k2l.end()){
		// already exists, bring to front
		ll.splice(ll.begin(),ll,founditr->second);
		// update value
		founditr->second->second = value; 
		return;
		}
		// key is not in cache, insert it in k2l and ll
		// create space if needed
		if(k2l.size()==cap){
			int delkey = ll.back().first;
			ll.pop_back();
			k2l.erase(delkey);
		}
		// insert new kv pair
		//emplace front has a variadic template, so whatever we pass here 
		//will be forwarded to the constructor of a pair in the same order
		ll.emplace_front(key,value);		
	}
};
```

## 206 Reverse Linked List

Reverse a singly linked list.

Hint:
A linked list can be reversed either iteratively or recursively. Could you implement both?
```cpp
// Linked list definition
struct ListNode{
	int val;
	ListNode* next;
	ListNode(int x): val(x), next(NULL){}
};
```
### Iterative solution
```cpp
class solution{
public:
	ListNode* reverseList(ListNode* head){
		ListNode* prev = NULL;
		ListNode*cur = head;
		while(cur && cur->next){
			// store cur->next for next iteration
			ListNode* temp = cur->next;
			cur->next = prev;
			prev = cur;
			cur = temp;
		}
		// last element i.e. new head
		if(cur) cur->next = prev;
		return cur; 
};
```



## 160 Intersection of Two Linked Lists
Write a program to find the node at which the intersection of two singly linked lists begins.

For example, the following two linked lists:

```
A:          a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗            
B:     b1 → b2 → b3
```
begin to intersect at node c1.

Notes:

If the two linked lists have no intersection at all, return `null`.
The linked lists must retain their original structure after the function returns.
You may assume there are no cycles anywhere in the entire linked structure.
Your code should preferably run in `O(n)` time and use only `O(1)` memory.

```cpp
// Linked List definition
struct ListNode {
	int val;
	ListNode* next;
	ListNode(int x): val(x), next(NULL) {}
};

class Solution{
	ListNode* getIntersectionNode(ListNode* headA, ListNode* headB){
	ListNode* cur1 = headA;
	ListNode* cur2 = headB;
	while(cur1!=cur2){
		cur1 = cur1?cur1->next:headA;
		cur2 = cur2?cur2->next->next:headB; 
	}
	return cur1;
	}
};
```

## 122 Best Time to Buy and Sell Stock II
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

```cpp
class Solution{
	int maxProfit(vector<int>& prices){
	int ret = 0;
	for(size_t p=1; p<prices.size();++p)
	// for seq a,b,c; a-b + b-c = a-c
	// so we simply accumulate successive differences
	ret += max(prices[p]-prices[p-1],0)
}
};
```

## 53 Maximum Subarray

Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

For example, given the array `[-2,1,-3,4,-1,2,1,-5,4]`,
the contiguous subarray `[4,-1,2,1]` has the largest sum = `6`.

click to show more practice.

More practice:
If you have figured out the `O(n)` solution, try coding another solution using the divide and conquer approach, which is more subtle.

```cpp
class Solution{
public:
	int maxSubArray(vector<int> A){
		int n = A.size();
		int ans = A[0],sum = 0;
		for(int i=0;i<n;i++){
			// accumulate
			sum += A[i];
			// include this element if it improves current ans (or break off seq)
			ans = max(sum,ans);
			// reset sum if it goes below 0
			sum = max(sum,0); 
		}
		return ans;
	}
}
```

## 69 Sqrt(x)

Implement `int sqrt(int x)`.

Compute and return the square root of `x`.

```cpp
class Solution{
public:
	int mySqrt(int x){
		return (int) pow(x,0.5);
	}	
};
```

Alternative techniques

1. Binary Search
2. Newton-Ralphson (also known as Heron's/Babylonian method)

## 232 Implement Queue using Stacks
Implement the following operations of a queue using stacks.

`push(x)` -- Push element x to the back of queue.
`pop()` -- Removes the element from in front of queue.
`peek()`-- Get the front element.
`empty()` -- Return whether the queue is empty.
Notes:

 - You must use only standard operations of a stack -- which means only `push to top`, `peek/pop from top`, `size`, and `is empty` operations are valid. 
 - Depending on your language, stack may not be supported natively. You may simulate a stack by using a list or deque (double-ended queue), as long as you use only standard operations of a stack. 
 - You may assume that all operations are valid (for example,
   no pop or peek operations will be called on an empty queue).

```cpp
class MyQueue{
	// Use 2 stacks, one with size cap and other bottomless
	stack<int> s1;
	stack<int> s2;
	int s1cap;
public:
	// constructor
	MyQueue(){
		stack<int> s1;
		stayck<int> s2;
		s1cap = 256;	 
	 }
	
	// push into queue
	void push(int x){
		s1.push(x); // push into s1
		if(s1.size()>=s1cap){
			// if s1 is full, empty it into s2
			while(!s1.empty()){
				s2.push(s1.top());
				s1.pop();
			}
		}
	}
	// pop from front of queue
	int pop(){
		// if s2 is empty transfer s1 to s2
		if(s2.empty()){
			while(!s1.empty()){
				s2.push(s1.top());
				s1.pop();
			}
		}
		// pop from s2
		int x = s2.top();
		s2.pop();
		return x;
	}
	// get the front element
	int peek(){
		// if s2 is empty transfer s1 to s2
		if(s2.empty()){
			while(!s1.empty()){
				s2.push(s1.top());
				s1.pop();
			}
		}
		return s2.top();
	}
	int empty(){
		return s1.empty() && s2.empty();
	}
};
```

## 386 Lexicographical Numbers
Given an integer `n`, return `1 - n` in lexicographical order.

For example, given 13, return: `[1,10,11,12,13,2,3,4,5,6,7,8,9]`.

Please optimize your algorithm to use less time and space. The input size may be as large as `5,000,000`.

```cpp
#include<stdlib.h>
class Solution {
private:
	// given a prefix, construct all extensions of it < n and store them in res (passed by reference) 
	void prefix_nums(int prefix, int n, vector<int>& res){
		for(int i=0;i<=9;i++){
			// append i to the prefix at the 10^0 place
			if(num>n)
			return;
			else{
				res.push_back(num);
				prefix_nums(num,n,res);
			}
		}
	}
}

public:
// since we want numbers starting from 1, the top level of recursion is performed here (notice for loop variables)
	vector<int> lexicalOrder(int n){
		vector<int> res = {};
		for(int num=1;num<9;num++){
			if(num>n) return res;
			else{
				// store the number
				res.push_back
				// rest with this number as prefix
				prefix_nums(num,n,res);
			}
		}
		return res;
	}
```

## 50 Pow(x, n)

Implement pow(x, n).

```cpp
class Solution {
public:
	// uses recursion, divide & conquer (sol(x,n) = f(sol(x,n/2)))
	// recursion depth logarithmic in n 
	double myPow(double x, int n){
		if(n==0) return 1;
		double temp = myPow(x,n/2);
		// even power, simply compute temp^2
		if(n%2==0) return (double)temp*temp;
		else{
		// odd power: temp*temp*x (positive power)
		if(n>0) return (double)(x*temp*temp);
		// temp*temp/x (negative power)
		else (temp*temp)/x;
		}
	}
}
```

## 56 Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

For example,
Given `[1,3],[2,6],[8,10],[15,18]`,
return `[1,6],[8,10],[15,18]`

```cpp
// Definition of interval struct
struct Interval {
	int start;
	int end;
	// overloaded constructors (can use default args instead too)
	Interval(): start(0), end(0) {}
	Interval(int s, int e): start(s), end(e) {}
}
```

```cpp
// custom compare function to sort vector of intervals
// ascending based on starting point
bool compare_intervals(Interval x, Interval y){return x.start < x.end}

class Solution(){
	vector<Interval> merge(vector<Interval>& intervals){
		sort(intervals.begin(),intervals.end(),compare_intervals);
		vector<Interval> res = {};
		for(int i=0;i<intervals.size();i++){
			if(res.size()>0){
				// test if last interval in res has overlap 
				// with intervals[i];
				if(intervals[i].start <= res[res.size()-1].end) // overlap: merge
				res[res.size()-1].end = max(res[res.size()-1].end,intervals[i].end);
				else
				res.push_back(intervals[i]); // no overlap, push direct
			}
			else
			res.push_back(intervals[i]);
		}
		return res;
	}
}

```

## 20 Valid Parenthesis
Given a string containing just the characters `'(', ')', '{', '}', '[' and ']',` determine if the input string is valid.

The brackets must close in the correct order, `"()"` and `"()[]{}"` are all valid but `"(]"` and `"([)]"` are not.

```cpp
class Solution() {
public:
	// valid parenthesis similar to valid nested function calls, 
	// which means we can use something like a call stack: 
	// 1) push opening brackets in stack
	// 2) if we see a closing bracket, pop from stack and check match
	bool isValid(string s){
		stack<char> parent;
		for(char& c: s){
			switch(c){
			case '(':
			case '[':
			case '{': parent.push(c); break; // common for all opening brackets
			// mismatched brackets: 
			case ')': 
				if(parent.empty() || parent.top()!='(') return false; 
				else parent.pop(); break;
			case ']': 
				if(parent.empty() || parent.top()!='[') return false; 
				else parent.pop(); break;
			case '}': 
				if(parent.empty() || parent.top()!='}') return false; 
				else parent.pop(); break;
			}
		}
		return parent.empty();//takes care of brackets with no close
	}	
}
```

## 287 Find the Duplicate Number

Given an array nums containing `n + 1` integers where each integer is between `1` and `n` (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.

**Note:**

 1. You must not modify the array (assume the array is read only). 
 2. You must use only constant, `O(1)` extra space. 
 3. Your runtime complexity should be less than $O(n^2)$. 
 4. There is only one duplicate number in the array, but it could be repeated more than once.

```cpp
// Use Floyd's cycle finding algorithm
// Assume that the array contents are like pointers to next location (base 1 indexing) 
class Solution {
	public:
	int findDuplicate(vector<int>& nums){
		if(nums.size()==2) return nums[0];//sureshoit duplicate
		if(nums.size()>2){
			int slow = nums[0];
			int fast = nums[nums[0]];
			while(slow!=fast){
				slow = nums[slow];
				fast = nums[nums[fast]];
			}
			// now both slow and fast are in the loop, increment slow from start
			slow = 0;
			while(slow!=fast){
				slow = nums[slow];
				fast = nums[fast];
			}
			return slow;
		}
		else
		return slow;
	}
}
```

## 2 Add Two Numbers
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Input**: `(2 -> 4 -> 3) + (5 -> 6 -> 4)`
**Output**: `7 -> 0 -> 8`

```cpp
// Definition of singly-linke list
struct ListNode {
	int val;
	ListNode* next;
	ListNode(int x): val(x), next(NULL) {}
}

class Solution {
public:
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2){
		ListNode* ptr1 = l1;
		ListNode* ptr2 = l2;
		int i = 0;
		ListNode* retptr = NULL;
		ListNode* prev = NULL;
		int carry = 0;
		while(ptr1 && ptr2){
			int x = ptr1->val + ptr2->val +carry;
			carry = x/10;
			ListNode* ptr = new ListNode(x%10);
			if(prev) prev->next = ptr;
			else retptr = ptr;
			prev = ptr;
			ptr1 = ptr1->next;
			ptr2 = ptr2->next;
		}
		ListNode* rem_ptr = ptr1?ptr1:ptr2;
		while(rem_ptr){
			int x = rem_ptr->val + carry;
			carry = x/10;
			ListNode* ptr = new ListNode(x%10);
			if(prev) prev->next = ptr;
			else retptr = ptr;
			prev = ptr;
			rem_ptr = rem_ptr->next;
		}
		if(carry){
			ListNode* ptr = new ListNode(1);
			if(prev) prev->next = ptr;
			else retptr = ptr;
		}
		return retptr;
	}
};
```

## 268 Missing Number
Given an array containing `n` distinct numbers taken from `0, 1, 2, ..., n`, find the one that is missing from the array.

For example,
Given nums = `[0, 1, 3]` return 2.

Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?

```cpp
class Solution {
	int missingNumber(vector<int>& nums){
	int x = 0;
	int y = 0;
	for(int i=0;i<n;i++){
		x+=nums[i];
		y+=i+1;
	}
	return y-x;
}
```

#7 Reverse Integer

Reverse digits of an integer.

Example1: x = 123, return 321
Example2: x = -123, return -321

Note:
The input is assumed to be a 32-bit signed integer. Your function should return 0 when the reversed integer overflows.

```cpp
class Solution {
public:
	int reverse(int x){
		long long int y = 0; // to store result
		int sign = x<0?-1,1; // to store result's sign
		// to store leftover number after division by powers of 10
		long long int z = x; 
		z = (sign==-1)?-z:z; // divide successively by this
		vector<int> digits = {}; // store digits of x;
		long long int b = 1;
		while(x/b != 0){
			int rem = x%10; // digit
			z = (z-rem)/10; // leftover number
			digits.push_back(rem);
			b*=10; 
		}
		// now compute reverse from digit vector
		b = 1;
		// traverse msb to lsb
		for(int i = digits.size()-1;i>=0;i++){
			y+=digits[i]*b;
			if(y>INT_MAX || Y<INT_MIN) return 0;
			b*=10;
		}
		return sign<0?-y:y;
	}
};
```

## 98 Validate Binary Search Tree

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
**Example 1:**
```
    2
   / \
  1   3
```
Binary tree `[2,1,3]`, return **true**.
**Example 2:**
```
    1
   / \
  2   3
```
Binary tree `[1,2,3]`, return **false**.

```cpp
// Definition of a binary tree node
struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x): val(x), left(NULL), right(NULL) {}
}
```

 **Note:** Use `long` instead of `int` for `low` and `high` if `INT_MAX` and `INT_MIN` can themselves be values in the tree 

```cpp
class Solution {
bool isValidBSTUtil(TreeNode* root, int low, int high){
	if(!root) return true;
	if(root->left && root->right)
		return root->left->val<root->val && root->left->val>low && root->right->val>root->val && rotyt->right->val<high && isValidBSTUtil(root->left,low,root->left) && isValidBSTUtil(root->right,root->val,high);
	else if(root->left) return root->left->val<root->val && root->left->val>low && isValidBSTUtil(root->left,low,root->left);
	else if(root->right) isValidBSTUtil(root->left,low,root->left) && isValidBSTUtil(root->right,root->val,high);
	return true;
}
public:
	bool isValidBST(TreeNode* root){
		return isValidBSTUtil(root, INT_MIN, INT_MAX);
	}
}
```


## 151 Reverse Words in a String

Given an input string, reverse the string word by word.

For example,
Given s = `"the sky is blue"`,
return `"blue is sky the"`.

Update (2015-02-12):
For C programmers: Try to solve it in-place in O(1) space.

```cpp
class Solution {
public:
	void reverseWords(string& s){
	string result("");
	int i = 0;// for indexing s
	while(i<s.size()){
	if(s[i]==' ') i++; // skip spaces
	else{
		int j = i+1;
		while(s[j]!=' ') j++;
		// prepend to result (for first word, no space inbetween)
		if(result.size()>0)
		result = s.substr(i,j-i) + " " + result;
		else
		result = s.substr(i,j-1) + result;
		i = j;
		}
		s = result;
	}
};
```

## 215 Kth Largest Element in an Array
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

For example,
Given `[3,2,1,5,6,4]` and `k = 2`, return `5`.

Note: 
You may assume k is always valid, 1 ≤ k ≤ array's length.

```cpp
class Solution {
public:
	// Use quickselect
	// randomized pivoting gives 12ms runtime vs 33ms of deterministic
	int partition(vector<int>& nums, int low, int high){
	    int pivot = low + rand() % (high-low);
	    swap(nums[high],nums[pivot]);
		int j = low;
		for(int i =low;i<high;i++){
		if(nums[i]>nums[high]){
			swap(nums[i],nums[j]);
			j++;
			}
		}
		swap(nums[j],nums[high]);
		return j;
	}
	int select(vector<int> nums, int low, int high, int k){
		if(low == high) return nums[low];
		int p = partition(nums,low,high);
		if(p+1==k) return nums[p];
		else if(k<p+1) return select(nums,low,p-1,k);
		else return select(nums,p+1,high,k);
	}
	int findKthLargest(vector<int>& nums, int k){
		return select(nums,0,nums.size()-1,k);
	}
};
```


## 88 Merge Sorted Array
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.

```cpp
class Solution {
public:
	void merge(vector<int>& nums1, int m, vector<int>& nums2, int n){
		int i = m-1, j = n-1, tar = m+n-1;
		while(j>=0){
			// following line takes care of (in that order):
			// 1. decrementing tar every iteration
			// 2. Not changing nums[tar] if we have exhausted nums2
			// 3. dumping appropriate element of nums1/nums2 in nums1 and updating
			//    i & j based on which one was dumped  
			nums1[tar--]=i>=0 && nums[i] > nums[j] ? nums1[i--]:nums2[j--];
		}
	}
}
```


## 49 Group Anagrams
Given an array of strings, group anagrams together.

For example, given: `["eat", "tea", "tan", "ate", "nat", "bat"]`, 
Return:
```
[
  ["ate", "eat","tea"],
  ["nat","tan"],
  ["bat"]
]
```


```cpp
class Solution {
	public:
		vector<vector<string>> groupAnagrams(vector<string>& strs){
			unordered_map<string,multiset<string>> mp;
			for(auto s:strs){
				string t = strSort(s);
				// Note: this multiset behaves like 
				// unordered_map<string,unordered_set<string>>
				mp[t].insert(s);
			}
			vector<vector<string>> anagrams;
			for(auto m:mp){
				vector<string> anagram(m.second.begin(),m.second.end());
				anagrams.push_back(anagram);
			}
			return anagrams;
		}
	private:
		string strSort(string& s){
			// uses counting sort to sort in time linear in the length
			int count = 26;
			for(int i=0;i<n;i++)
			count[s[i]-'a']++;
			int p=0; // current pointer
			string t(n,'a');// sorted result
			for(int j = 0; j < 26;j++) // dump all identical chars at once to t
				for(int i = 0;i < count[j];i++)
				t[p++]=j;
			return t;
		}
		
}
```

## 102. Binary Tree Level Order Traversal

Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,
```
    3
   / \
  9  20
    /  \
   15   7
```
return its level order traversal as:
```
[
  [3],
  [9,20],
  [15,7]
]
```

```cpp
class Solution {
public:
	vector<vector<int>> levelOrder(TreeNode* root){
		vector<vector<int>> res = {};
		if(!root) return res;
		vector<int> level_vals = {root->val};
		vector<TreeNode*> level_nodes = {root};
		while(level_vals.size()!=0){
			res.push_back(level_vals);
			vector<int> temp_vals = {};
			vector<TreeNode*> temp_nodes = {};
			for(int i=0;i<level_nodes.size();i++){
				if(level_nodes[i]->left){
					temp_vals.push_back(level_nodes[i]->left->val);
					temp_nodes.push_back(level_nodes[i]->left)
				}
				if(level_nodes[i]->right){
					temp_vals.push_back(level_nodes[i]->right->val);
					temp_nodes.push_back(level_nodes[i]->right)
				}
			}
			level_nodes = temp_nodes;
			level_vals = temp_vals;
		}
		return res;
	}
};
```

## 225. Implement Stack using Queues

Implement the following operations of a stack using queues.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
empty() -- Return whether the stack is empty.
**Notes:**

 1. You must use only standard operations of a queue -- which means only `push to back`, `peek/pop from front`, `size`, and `is empty` operations are valid. 
 2. Depending on your language, queue may not be supported natively. You may simulate a queue by using a list or  deque (double-ended queue), as long as you use only standard operations of a queue. 
 3. You may assume that all operations are valid (for example, no pop or top operations will be called on an empty stack).

```cpp
class MyStack{
private:
	queue<int> q1;
	queue<int> q2;
public:
	// constructor
	Mystack() {q1 = {}; q2 = {};}
	
	// push into q2, empty q1 to q2, reverse roles
	// so q2 is always empty at the end of this
	// thus it is used to make sure the latest element remains at the front of q1
	void push(int x){
		q2.push(x);
		while(q1.empty()){
			q2.push(q1.front());
			q1.pop();
		}
		swap(q1,q2);
	}
	
	// just pop from q1
	int pop(){
		int x = q1.front();
		q1.pop();
		return x;
	}
	
	int top(){
		int x = q1.front();
		return x;
	}
	bool empty(){
		return q1.empty() && q2.empty();
	};
```

## 3. Longest Substring Without Repeating Characters

Given a string, find the length of the longest substring without repeating characters.

Examples:

Given `"abcabcbb"`, the answer is `"abc"`, which the length is 3.

Given `"bbbbb"`, the answer is `"b"`, with the length of 1.

Given `"pwwkew"`, the answer is `"wke"`, with the length of 3. Note that the answer must be a substring, `"pwke"` is a subsequence and not a substring. 

```cpp
class Solution {
public:
	int lengthOfLongestSubstring(string s){
		int n = s.size();
		int i=0,j=0;
		int maxlen = 0;
		unordered_set<int> seen = {}; // current buffer
		// j is the leading pointer while i is the trailing pointer
		// if we bump into a preseen character, we
		// increment i, while erasing char pointed to by i
		// until we hit the previous occurance of char at j 
		while(j<n){
			if(seen.find(s[j])!=seen.end()){
				// found preseen char at j
				maxlen = max(maxlen,j-i);
				// increment i until we hit the first occurance
				// of the duplicate in our current string
				while(s[i]!=s[j]){
					if(seen.find(s[i])!=seen.end())
					seen.erase(s[i]);
					i++;
				}
				i++; // point i beyond the first occurance 
				j++;
			}
			else
			seen.insert(s[j++]); // grow from front
		}
		// consider last non-repeating str
		maxlen = max(maxlen,n-i);
		return maxlen;
	}
};
```

## 103 Binary Tree Zigzag Level Order Traversal

Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree `[3,9,20,null,null,15,7]`,
```
    3
   / \
  9  20
    /  \
   15   7
```
return its zigzag level order traversal as:
```
[
  [3],
  [20,9],
  [15,7]
]
```

```cpp
class Solution {
	vector<vector<int>> zigzagLevelOrder(TreeNode* root){
		vector<vector<int>> res = {};
		if(!root) return res;
		vector<int> level_vals = {root->val}
		vector<TreeNode*> level_nodes = {root};
		bool state = true;
		while(level_nodes.size()!=0){
			if(!state) reverse(level_vals.begin(),level_vals.end());
			res.push_back(lavel_vals);
			state = !state;
			vector<int> temp_vals = {};
			vector<int> temp_nodes = {};
			for(int i=0;i<level_nodes.size();i++){
				if(level_nodes[i]->left){
					temp_vals.push_back(level_nodes[i]->left->val);
					temp_nodes.push_back(level_nodes[i]->left);
				}
				if(level_nodes[i]->right){
					temp_vals.push_back(level_nodes[i]->right->val);
					temp_nodes.push_back(level_nodes[i]->right);
				}
			}
			level_vals = temp_vals;
			level_nodes = temp_nodes;
		}
		return res; 
	}
};
```

## 26. Remove Duplicates from Sorted Array

Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

For example,
Given input array nums = `[1,1,2]`,

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the new length.

```cpp
class Solution {
public:
	int removeDuplicates(vector<int>& nums){
		int j=0; // current pointer
		int i=0; // last unique pointer
		while(j<nums.size()){
			vector<int>::iterator ub;
			ub = upper_bound(nums.begin()+j+1,nums.end(),nums[j]);
			j = ub-nums.begin();
			nums[i++] = nums[j-1];
		}
		return i;
	}	
}
```

## 105. Construct Binary Tree from Preorder and Inorder Traversal

Given preorder and inorder traversal of a tree, construct the binary tree.
**Note:**
You may assume that duplicates do not exist in the tree.

```cpp
class Solution {
	TreeNode* buildTreeUtil(vector<int>& preorder, vector<int>& inorder, int ps, int pe, int is, int ie){
		if(ps < pe) return NULL;
		// value at preorder[ps] is root
		TreeNode* root_node = new TreeNode(preorder[ps]);
		int rloc = -1;
		// find preorder[ps] index rloc in inorder[is..ie], to get size of left subtree 
		// = rloc-is, thus ltree = (preorder[ps+1..ps+(rloc-is)]) and beyond is 
		// right subtree (preorder[ps+(rloc-is)..pe]) 
		for(int i=is; i<=ie;i++){
			if(inorder[i]==root_node->val){
				rloc = i;
				break;
			}
		}
		root->left = buildTreeUtil(preorder,inorder,ps+1,ps+(rloc-is),is,rloc-1);
		root->right = buildTreeUtil(preorder,inorder,ps+(rloc-is)+1,rloc+1,ie);
		return root_node;
	}
}
public:
	TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder){
		return buildTreeUtil(preorder, inorder,0,preorder.size()-1,0,inorder.size()-1);
}
```


## 79. Word Search

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

For example,
Given board =
```
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
```

word = `"ABCCED"`, -> returns `true`,
word = `"SEE"`, -> returns `true`,
word = `"ABCB"`, -> returns `false`.

```cpp
class Solution {
private:
	bool hasstr(vector<vector<char>> board, string word, int ws, int x, int y){
		if(x<0||x>=m||y<0||y>=n||board[x][y]=='\0'||word[ws]!=board[x][y])
		return false;
		if(ws+1 == word.size()) return true;
		char t = board[x][y];
		board[x][y]='\0';
		if(hasstr(board,word,ws+1,x+1,y) || hasstr(board,word,ws+1,x-1,y) || hasstr(board,word,ws+1,x,y+1) || hasstr(board,word,ws+1,x-1,y-1)) return true;
		board[x][y] = t;
		return false;
	}
public:
	bool exist(vector<vector<char>>& board, string word){
		m = board.size();
		n = board[0].size();
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++)
			if(hasstr(board,word,0,i,j)) return true;
		}
		return false;
	}
}
```

## 141. Linked List Cycle

Given a linked list, determine if it has a cycle in it.

Follow up:
Can you solve it without using extra space?

```cpp
// Floyd's cycle finding
class Solution{
public:
	hasCycle(ListNode* head){
		if(head==NULL) return false;
		ListNode* slow = head;
		ListNode* fast = head;
		while(fast->next && fast->next->next){
			slow = slow->next;
			fast = fast->next;
			if(slow==fast) return true;
		}
		return false;
	}
}
```

## 5. Longest Palindromic Substring

Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

**Example:**
```
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer
```
**Example:**

```
Input: "cbbd"

Output: "bb"
```

```cpp
class Solution {
public:
	string longestPalindrome(string s){
		if(s.empty()) return "";
		if(s.size()==1) return s;
		int min_start = 0, max_len = 1;
		for(int i=0;i<s.size();){
			if(s.size()-i<=max_len/2) break;//no chance of getting better
			int j=i,k=i; // k is leading ptr while j is trailing
			// skip identical
			while(k<s.size()-1 && s[k+1]==s[k]) k++;
			// expand k to right and j to left
			while(k<s.size()-1 && j>0 s[k+1]==s[j-1]) {++k;++j;}
			int new_len = k-j+1;
			if(new_len>max_len) {min_start = j; max_len = new_len;}
		}
		return s.substr(min_start,max_len);
	}
}
```

## 63. Unique Paths II
Follow up for "Unique Paths" (62):

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and empty space is marked as `1` and `0` respectively in the grid.

For example,
There is one obstacle in the middle of a 3x3 grid as illustrated below.
```
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
```
The total number of unique paths is `2`.

**Note**: m and n will be at most `100`.

```cpp
class Solution {
string idx2key(int m, int n){
	return to_string(m)+"_"+to_string(n);
}
public:
	unordered_map<string,int> cache;
	Solution(){cache = {};}
	int uniquePathsUtil(vecor<vector<int>>& obstacleGrid, int m, int n){
		if(m == obstacleGrid.size()-1 && n==obstacleGrid[0]/size()-1)
		return 1;
		else{
			if(cache.find(idx2key(m,n))!=cache.end()) return cache(idx2key(m,n));
			cache[idx2key(m,n)] = 0;
			cache[idx2key(m,n)]+= m+1<obstaclGrid.size() && obstacleGrid[m+1][n]!=1 ? uniquePathsUtil(obstaclGrid,m+1,n): 0;
			cache[idx2key(m,n)]+= n+1<obstaclGrid[0].size() && obstacleGrid[m][n+1]!=1 ? uniquePathsUtil(obstaclGrid,m,n+1): 0;
			return  cache[idx2key(m,n)];p
		}
	}
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
		if(obstacleGrid[obstacleGrid.size()-1][obstacleGrid[0].size()-1]==1||
		obstacleGrid[0][0]==1) return 0;
		return uniquePathsUtil(obstacleGrid,0,0);
	}
}
```

## 13. Roman to Integer

Given a roman numeral, convert it to an integer.

Input is guaranteed to be within the range from 1 to 3999.

**Notes:**
canonical numbers (numbers described by a single letter):
I = 1
V = 5
X = 10
L = 50
C = 100
D = 500
M = 1000

Additive rule: Use left to right descending value canonical numbers to represent number
e.g. XVII = 17

Subtractive rule: In case additive rule returns more than 4 same characters in a row, write next larger canonical numeral and prefix numeral sequence to subtract.
e.g. IIII = 4 is written as IV (5-1)

```cpp
class Solution{
public:
	int romanToInt(string s);
	unordered_map<char,int> valmap = {{'I',1},{'V',5},{'X',10},{'L',50},{'C',100},{'D',500},{'M',1000}};
	if(s.size()==1) return valmap[s[0]];
	else if(s.size()==0) return 0;
	int maxpos = 0;
	int maxnum = 0;
	// find maximum value char
	for(int i=0;i<s.size();i++){
		if(maxnum<valmap[s[i]]){
			maxnum = valmap[s[i]];
			maxpos = i;
		}
	}
	// valmap[s[maxpos]] - subtractive-part + additive-part
	return valmap[s[maxpos]] - romanToInt(s.substr(0,maxpos)) + romanToInt(s.substr(maxpos+1,s.size()-maxpos));
}
```

## 100. Same Tree
Given two binary trees, write a function to check if they are equal or not.

Two binary trees are considered equal if they are structurally identical and the nodes have the same value.

```cpp
class Solution {
public:
	bool isSameTree(TreeNode* p, TreeNode* q){
		if(p && q){
			if(p->val == q->val) return isSameTree(p->left,q->left) && isSameTree(p->right,q->right);
		}
		if(p==NULL && q==NULL) return true;
		return false;
	}
}
```

## 139. Word Break

Given a **non-empty** string s and a dictionary wordDict containing a list of **non-empty** words, determine if s can be segmented into a space-separated sequence of one or more dictionary words. You may assume the dictionary does not contain duplicate words.

For example, given
s = `"leetcode"`,
dict = `["leet", "code"]`.

Return true because `"leetcode"` can be segmented as `"leet code"`.

UPDATE (2017/1/4):
The wordDict parameter had been changed to a list of strings (instead of a set of strings). Please reload the code definition to get the latest changes.

```cpp
class Solution {
public:
	bool wordBreakUtil(string s, unordered_set<string>& wordSet, unordered_set<string>& badstr){
		if(s.size()==0) return true;
		int len = 1;
		while(len<=s.size()){
			string seg = s.substr(0,len);
			if(wordSet.find(seg)!= wordset.end()){
				string suffix = badstr.find(s.substr(len,s.size()-len));
				if(badstr.find(suffix)==badstr.end() && wordBreakUtil(suffix, wordSet, badstr))
				return true;
				else
				badstr.insert(suffix);
				}
			}
			len++;
		}
	}
boolwordBreak(string s, vector<string>& wordDict){
	unordered_set<string> wordSet = {};
	for(int i=0;i<wordDict.size();i++) wordset.insert(wordDict[i]);
	unordered_set<string> badstr = {};
	return wordBreakUtil(s,wordSet,badstr);
}
}
```


## 15. 3Sum

Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note: The solution set must not contain duplicate triplets.
```js
For example, given array S = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

```
class Solution {
public:
vector<vector<int>> threeSum(vector<int>& num){
	vector<vector<int>> res= {};
	sort(num.begin(),num.end());
	for(int i=0;i<num.size();i++){
		int target = -num[i];
		int front = i+1;
		int back = num.size()-1;
		while(front<back){
			int sum = num[front]+num[back];
			if(sum<target) front++;
			else if(sum>target) back++;
			else{
				vector<int> triplet(3,0);
				triplet[0] = num[i];
				triplet[1] = num[front];
				triplet[2] = num[back];
				res.push_back(triplet);
				while(front<back && num[front]==triplet[1]) front++; // skip duplicates of num[front]
				while(front<back && num[back]==triplet[2]) back--; // skip duplicates of num[back]
			}
		}
		while(i+1<num.size() && num[i+1]==num[i]) i++;
	}
	return res;
}
}
```


## 101. Symmetric Tree
Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree `[1,2,2,3,4,4,3]` is symmetric:

```
    1
   / \
  2   2
 / \ / \
3  4 4  3
```
But the following `[1,2,2,null,3,null,3]` is not:
```
    1
   / \
  2   2
   \   \
   3    3
```
Note:
Bonus points if you could solve it both recursively and iteratively.

### Recursive Solution

```cpp
class Solution {
	bool areMirrored(TreeNode* x, TreeeNode* y){
		if(!x && !y){
			return true;
		if(x && y) return x->val == y->val && areMirrored(x->left,y->right) && areMirrored(x->right, y->left);
		}
	}
}

public:
	bool isSymmetric(TreeNode* root){
		if(!root) return true;
		return areMirrored(root->left,root->right);
	}
```

# 172. Factorial Trailing Zeroes
Given an integer n, return the number of trailing zeroes in n!.

**Note:** Your solution should be in logarithmic time complexity.

**Explanation:**
A trailing 0 occurs in n! as many times as there are numbers ending in 5 in [1..n] recursively i.e.
$Z(n) = Z(\frac{n}{5}) + \frac{n}{5}$
(division is understood to be integer) 


```cpp
class Solution {
public:
	int trailingZeroes(int n){
		int sum = 0;
		int tmp = 0;
		// we determine how many multiples of 5^x occur in the range 1..n, 1 = 1.., and add them together
		while(n/5>0){
			tmp = n/5;
			sum+=tmp;
			n = tmp;
		}
		return sum;
	}
}
```

## 131. Palindrome Partitioning

Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.

For example, given s = `"aab"`,
Return
```
[
  ["aa","b"],
  ["a","a","b"]
]
```

```cpp
class Solution{
public:
	bool isPalindrome(string s){
		int start = 0;
		int end = s.size()-1;
		while(start<=end){
			if(s[start]!=s[end]) return false;
			start++; end--;
		}
		return true;
	}
	vector<vector<string>> partitionUtil(string s, unordered_map<string, vector<string>>& cache){
		if(s.size()==0) return {{}};
		else if(s.size()==1) return {{string(1,s[0])}};
		// search in cache
		if(cache.find(s)!= cache.end()) return cache(s);
		vector<vector<string>> allparts = {};
		// iterate through all possible pivot points
		for(int i=1;i<s.size()+1;i++){
			string left = s.substr(0,i);
			if(isPalindrome(left)){
				string right = s.substr(i,s.size()-i);
				vector<vector<string>> parts = {};
				vector<vector<string>> rightparts = partitionUtil(right, cache);
				for(int k=0;k<rightparts.size();k++){
					vector<string> part = {left};
					part.reserve(part.size()+rightparts[k].size());
					part.insert(part.end(),rightparts[k].begin(),
					rightparts[k].end());
					parts.push_back(part);
				}
				allparts.reserve(allparts.size()+parts.size());
				allparts.insert(allparts.end(),parts.begin(),parts.end());
			}
		}
		cache[s] = allparts;
		return allparts;
	}
};
```

## 24. Swap Nodes in Pairs

Given a linked list, swap every two adjacent nodes and return its head.

For example,
Given `1->2->3->4`, you should return the list as `2->1->4->3`.

Your algorithm should use only constant space. You may not **strong text**modify the values in the list, only nodes itself can be changed.

```cpp
class Solution {
	ListNode* swapPairs(ListNode* head){
		if(head==NULL || head->next==NULL) return head;
		ListNode* temp = head->next->next;
		ListNode* new_head = head->next;
		head->next->next = head;
		head->next = swapPairs(temp);
		return new_head;
	}
}
```

## 230. Kth Smallest Element in a BST

Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note: 
You may assume k is always valid, 1 ? k ? BST's total elements.

Follow up:
What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?

```cpp
	int kthSmallestUtil(TreeeNode* root,int k,int& val){
		if(!root) return 0;
		int lsize = kthSmallestUtil(root->left,k,val){
			// kth found here
			if(val!=-1) return 0;
			if(lsize()==k-1) {val = root->val; return 0;}
			int rsize = kthSmallestUtil(root->right, k-lsize-1,val);
			if(val!=-1) return 0;
			return lsize+rsize-1;
		}
public:
	int kthSmallest(TreeNode* root, int k){
		int val = -1;
		int s = kthSmallestUtil(root, k, val);
		return val;
	}	
};
```

## 16. 3Sum Closest

Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

    For example, given array S = {-1 2 1 -4}, and target = 1.

    The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

```cpp
class Solution {
public:
	int threeSumClosest(vector<int>& num, int target){
		int min_dist = INT_MAX;
		int min_dist_sum = 0;
		sort(num.begin(),num.end());
		for(int i=0;i<num.size();i++){
			int target2 = target - num[i];
			int front = i+1;
			int back = num.size()-1;
			while(front<back){
				int sum = num[front]+num[back];
				if(sum<target2){
					if(min_dist>target2-sum){
						min_dist = target2-sum;
						min_dist_sum = sum+num[i];
					}
					front++;
				}
				else if(sum>target2){
					if(min_dist>sum-target2){
						min_dist =target2-sum;
						min_dist_sum = sum+num[i];
					}
					back--;
				}
				else
				return target; 
			}
		}	
	}
	return min_dist_sum;
};
```

## 274. H-Index

Given an array of citations (each citation is a non-negative integer) of a researcher, write a function to compute the researcher's h-index.

According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have at least h citations each, and the other N − h papers have no more than h citations each."

For example, given `citations = [3, 0, 6, 1, 5]`, which means the researcher has `5` papers in total and each of them had received `3, 0, 6, 1, 5` citations respectively. Since the researcher has `3` papers with at least `3` citations each and the remaining two with no more than `3` citations each, his h-index is `3`.

Note: If there are several possible values for `h`, the maximum one is taken as the `h`-index.


```cpp
class Solution {
	int hIndex(vector<int> citations){
		sort(citations.begin(),citations.end());
		for(int i=0;i<citations.size();i++){
			if(citations[i]>=citations.size()-i) return citations.size()-i;
		}
		return 0;
	}
}
```

## 11. Container With Most Water

Given n non-negative integers $a_1, a_2,..., a_n$, where each represents a point at coordinate $(i, a_i)$. n vertical lines are drawn such that the two endpoints of line $i$ is at $(i, a_i)$ and $(i, 0)$. Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

```cpp
class Solution {
public:
	int maxArea(vector<int>& height){
		int water = 0;
		int i=0, j = height.size()-1;
		while(i<j){
			int h = min(height[i],height[j]);
			water = max(water,(j-i)*h);
			while(height[i]<=h && i<j) i++;
			while(height[j]<=h && i<j) j--;
		}
		return water;
	}
}
```

## 113. Path Sum II

Given a binary tree and a sum, find all root-to-leaf paths where each paths sum equals the given sum.

For example:
Given the below binary tree and `sum = 22`:

          5
         / \
        4   8
       /   / \
      11  13  4
     /  \    / \
    7    2  5   1

return

    [
       [5,4,11,2],
       [5,8,4,5]
    ]


```cpp
class Solution {
	vector<vector<int>> pathSum(TreeNode* root, int sum){
		if(!root) return {};
		vector<vector<int>> res = {};
		vector<vector<int>> leftpaths = {};
		vector<vector<int>> rightpaths = {};
		// left
		if(root->left)
		leftpaths = pathSum(root->left,sum-root->val);
		// right
		if(root->right)
		rightpaths = pathSum(root->right, sum-root->val);
		
		if(!root->left && !root->right && root->val = sum) res.push_back({root->val});
		for(int i=0;i<leftpaths.size();i++){
			vector<int> path = {root->val};
			for(int j=0;j<leftpaths[i].size();j++)
			path.push_back(leftpaths[i][j]);
			res.push_back(path);
		}
		for(int i=0;i<rightpaths.size();i++){
			vector<int> path = {root->val};
			for(int j=0;j<rightpaths[i].size();j++)
			path.push_back(rightpaths[i][j]);
			res.push_back(path);
		}
		return res;
	}
}
```

## 556. Next Greater Element III

Given a positive 32-bit integer n, you need to find the smallest 32-bit integer which has exactly the same digits existing in the integer n and is greater in value than n. If no such positive 32-bit integer exists, you need to return -1.
Example 1:

Input: 12
Output: 21

Example 2:

Input: 21
Output: -1

```cpp
class Solution {
public:
	int nextGreaterElement(int n){
		auto digits = to_string(n);
		next_permutation(begin(digits), end(digits));
		auto res = stoll(digits);
		return (res>INT_MAX || res <=n) ? -1:res;
	}
}
```


## 208. Implement Trie (Prefix Tree)


Implement a trie with `insert`, `search`, and `startsWith` methods.

Note:
You may assume that all inputs are consist of lowercase letters a-z.


```cpp
typedef struct TrieNode {
	unordered_map<char,TrieNode*> children;
	bool is_leaf;
	char c;
} TrieNode;

class Trie{
private: TrieNode* root;
public:
	Trie(){
		root = new TrieNode;
		root->children = {};
		root->is_leaf = true;
		root->c = '\0';
	}
	void delNode(TrieNode* t){
		if(t){
			for(auto elem: children){
				delNode(children.second);
			}
			delete t;
		}
	}
	~Trie(){delNode(root);}
	TrieNode* longestMatch(string str, int& len){
		if(str.size()==0){
			len = 0;
			return root;
		}
		TrieNode* current = root;
		int i=0;
		do{
			if(current->children.find(str[i])!current->children.end())
			current = current->children[str[i]];
			else
			break;
			i++;
		}while(i<str.size())
		len = i;
		return current;
	}

	void insert(string str){
		int len = -1;
		TrieNode* t = longestMatch(str,len);
		for(int i=len+1;i<str.size();i++){
			t->children[str[i-1]] = new TrieNode;
			t->children[[str-1]]->c = str[i-1];
			t->children[str[i-1]]->children = {};
			t->children[str[i-1]]->is_leaf = false;
			t = t->children[str[i-1]]; 
		}
		t->is_leaf = false;
	}
	bool search(string str){
		int len = -1;
		TrieNode* t = longestMatch(str,len);
		if(!t || len!=str.size() || !t->is_leaf) return false;
		return true;
	}
	bool startsWith(string str){
		int len = -1;
		TrieNode* t = longestMatch(str,len);
		if(len!=str.size()) return false;
		return true;
	}
};
```

## 266. Palindrome Permutation

Given a string, determine if a permutation of the string could form a palindrome.

For example,
`"code"`-> False, `"aab"` -> True, `"carerac"` -> True.

```cpp
class Solution {
public:
	bool canPermutePalindrome(string s){
		unordered_map<char,int> counts = {};
		for(int i=0;i<s.size();i++){
			if(counts.find(s[i])!=counts.end()) counts[s[i]] = 1;
			else counts[s[i]] += 1;
		}
		int nb_odd = 0;
		for(const auto& elem:counts){
			if(elem.second%2!=0) nb_odd++;
		}
		if(nb_odd<=1) return true;
		return false;
	}
}
```

## 110. Balanced Binary Tree

Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

```cpp
class Solution {
	int height(TreeNode* root){
		if(!root) return 0;
		else return max(height(root)->left,height(root->right)) +1;
	}
	bool isBalanced(TreeNode* root){
		if(!root) return true;
		else return isBalanced(root->left) && isBalanced(root->right) && abs(height(root->left)-height(root->right)) <=1;
	}
}
```


# NON-BLOOMBERG
## 123. Best Time to Buy and Sell Stock III

**Note**: This solution uses dynamic programming
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int states[2][4] = {INT_MIN, 0, INT_MIN, 0}; // 0: 1 buy, 1: one buy/sell, 2: 2 buys/1 sell, 3, 2 buys/sells
        int len = prices.size(), i, cur = 0, next =1;
        for(i=0; i<len; ++i)
        {
            states[next][0] = max(states[cur][0], -prices[i]);
            states[next][1] = max(states[cur][1], states[cur][0]+prices[i]);
            states[next][2] = max(states[cur][2], states[cur][1]-prices[i]);
            states[next][3] = max(states[cur][3], states[cur][2]+prices[i]);
            swap(next, cur);
        }
        return max(states[cur][1], states[cur][3]);
    }
};
```

## 323. Number of Connected Components in an Undirected Graph

Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to find the number of connected components in an undirected graph.

Example 1:
```
     0          3
     |          |
     1 --- 2    4
```
Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], return 2.

Example 2:
```
     0           4
     |           |
     1 --- 2 --- 3
```
Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [3, 4]], return 1.

Note:
You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

```cpp
class Solution {
public:
    void print_vec(vector<int> vec){
        for(int i=0;i<vec.size();i++){
            cout<< vec[i] << " ";
        }
        cout << endl;
    }
    
    int root(vector<int> id, int i){
        while (i != id[i]) 
        i = id[i];
        return i;
    }
    
    int root_pc(vector<int>& id,int i){
        while (i != id[i]) {
            id[i] = id[id[i]];
            i = id[i];
        }
        return i;
    }

    
    bool find(vector<int> id, int i, int j){
        return root(id,i)==root(id,j);
    }
    
    void unite(vector<int>& id, vector<int>& sz, int p, int q){
        int i = root(id,p);
        int j = root(id,q);
        if(sz[i]<sz[j]){
            id[i] = j; sz[j]+=sz[i];
        }
        else{
            id[j] = i; sz[i]+=sz[j];
        }    
    }
    
    int countComponents(int n, vector<pair<int, int>>& edges) {
        vector<int> id(n,0);
        vector<int> sz(n,1);
        iota(id.begin(), id.end(), 0);
        for(int i=0;i<edges.size();i++){
            //cout << edges[i].first << " " << edges[i].second << endl;
            if(root_pc(id,edges[i].first)!=root_pc(id,edges[i].second)){
                unite(id,sz,edges[i].first,edges[i].second);
            }
            //print_vec(id);
        }
        unordered_set<int> cc;
        for(int i=0;i<n;i++){
            cc.insert(root(id,id[i]));
        }
        return cc.size();
    }
};
```

## 4. Median of Two Sorted Arrays

There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

Example 1:
```
nums1 = [1, 3]
nums2 = [2]
```

The median is 2.0

Example 2:
```
nums1 = [1, 2]
nums2 = [3, 4]
```
The median is (2 + 3)/2 = 2.5

```cpp
class Solution {
public:
    int kth(vector<int>& a, int m, vector<int>& b, int n, int k) {
        
        if (m < n) return kth(b,n,a,m,k);
        if (n==0) return a[k-1];
        if (k==1) return min(a[0],b[0]);

        int j = min(n,k/2);
        int i = k-j;
        if (a[i-1] > b[j-1]) {
            vector<int> bx(b.begin()+j,b.end()); 
            return kth(a,i,bx ,n-j,k-j);
        }
        vector<int> ax(a.begin()+i,a.end());
        return kth(ax,m-i,b,j,k-i);
    }

    double findMedianSortedArrays(vector<int>& a, vector<int>& b) {
        int m = a.size();
        int n = b.size();
        int k = (m+n)/2;
        int m1 = kth(a,m,b,n,k+1);
        if ((m+n)%2==0) {
            int m2 = kth(a,m,b,n,k);
            return ((double)m1+m2)/2.0;
        }
        return m1;
    }
};
```

## 104. Maximum Depth of Binary Tree

Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
private:
    int maxDepth2(TreeNode* root, int maxsofar){
        int ldepth = 0;
        int rdepth = 0;
        if(root ==NULL)
        return 0;
        if((root->left)!=(TreeNode*)NULL)
        ldepth = maxDepth2(root->left,maxsofar);
        if((root->right)!=(TreeNode*)NULL)
        rdepth = maxDepth2(root->right,maxsofar);
        return max(maxsofar,max(ldepth+1,rdepth+1));
    }
    
public:
    int maxDepth(TreeNode* root) {
        return maxDepth2(root,0);
    }
};
```
## 237. Delete Node in a Linked List

Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.

Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3, the linked list should become 1 -> 2 -> 4 after calling your function.

Show Company Tags
Show Tags
Show Similar Problems


```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    void deleteNode(ListNode* node) {
        while(1){
            if(node->next!=NULL){
                if(node->next->next==NULL){
                    // secondlast node
                    node->val = node->next->val;
                    node->next = NULL;
                    return;
                }
                else{
                    // interim node
                    node->val = node->next->val;
                    node = node->next;
                }
            }
            else{
                return;
            }
        }
    }
};
```

## 170. Two Sum III - Data structure design

Design and implement a TwoSum class. It should support the following operations: add and find.

`add` - Add the number to an internal data structure.
`find` - Find if there exists any pair of numbers which sum is equal to the value.

For example,
```
add(1); add(3); add(5);
find(4) -> true
find(7) -> false
```

```cpp
class TwoSum {
private:
    unordered_map<int,int> numscount;
    vector<int> numsvec;
public:
    /** Initialize your data structure here. */
    TwoSum() {
        numscount = {};
        numsvec = {};
    }
    
    /** Add the number to an internal data structure.. */
    void add(int number) {
        if(numscount.find(number)==numscount.end()){
            numscount[number] = 1;
            numsvec.push_back(number);
        }
        else
            numscount[number]+=1;
    }
    
    /** Find if there exists any pair of numbers which sum is equal to the value. */
    bool find(int value) {
        for (int i = 0;i<numsvec.size();i++) {
            int x = value-numsvec[i];
            numscount[numsvec[i]]-=1; 
            if(numscount.find(x)!=numscount.end() && numscount[x]>=1){
                numscount[numsvec[i]]+=1;
                return true;
            }
            numscount[numsvec[i]]+=1;
        }
        return false;
    }
};

/**
 * Your TwoSum object will be instantiated and called as such:
 * TwoSum obj = new TwoSum();
 * obj.add(number);
 * bool param_2 = obj.find(value);
 */
```

## 405. Convert a Number to Hexadecimal

Given an integer, write an algorithm to convert it to hexadecimal. For negative integer, two’s complement method is used.

Note:

 1. All letters in hexadecimal (a-f) must be in lowercase. The hexadecimal string must not contain extra leading 0s.
 2. If the number is zero, it is represented by a single zero character '0'; otherwise, the first character in the hexadecimal string will not be the zero character. 
 3. The given number is guaranteed to fit within the range of a 32-bit signed integer. 
 4. You must not use any method provided by the library which converts/formats the number to hex directly.

Example 1:

```
Input:
26

Output:
"1a"

```

Example 2:
```
Input:
-1

Output:
"ffffffff"
```

```cpp
class Solution {
    
public:
    string digit2hex(int num){
        vector<string> dvec = {"0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f"};
        return dvec[num % 16];
    }
    string toHex(int snum) {
        if(snum == INT_MIN) return "80000000";
        else if(snum == 0) return "0";
        string ostr = ""; 
        unsigned int num = (unsigned int) snum;
        
        //cout << std::bitset<32>(num) << endl;
        int mask = 15;
        int i = 0;
        while(i<= 7 && (((~0)<<4*i) & num)!=0 ){
            //cout << ((mask & num) >> 4*i);
            ostr = digit2hex((mask & num) >> 4*i)+ostr;
            mask = mask << 4;
            i++;
        }
        return ostr;
    }
};
```

## 476. Number Complement

Given a positive integer, output its complement number. The complement strategy is to flip the bits of its binary representation.

Note:
The given integer is guaranteed to fit within the range of a 32-bit signed integer.
You could assume no leading zero bit in the integer’s binary representation.

Example 1:
```
Input: 5
Output: 2
```
Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.

Example 2:
```
Input: 1
Output: 0
```
Explanation: The binary representation of 1 is 1 (no leading zero bits), and its complement is 0. So you need to output 0.

```cpp
class Solution {
public:


    int findComplement( int num) {
        unsigned mask = ~0;
        while (num & mask) mask <<= 1;
        return (~num) & (~mask);
    }
};
```


## 458. Poor Pigs

There are 1000 buckets, one and only one of them contains poison, the rest are filled with water. They all look the same. If a pig drinks that poison it will die within 15 minutes. What is the minimum amount of pigs you need to figure out which bucket contains the poison within one hour.

Answer this question, and write an algorithm for the follow-up general case.

Follow-up:

If there are n buckets and a pig drinking poison will die within m minutes, how many pigs (x) you need to figure out the "poison" bucket within p minutes? There is exact one bucket with poison.

```cpp
class Solution {
public:
    int poorPigs(int buckets, int minutesToDie, int minutesToTest) {
        if(buckets<=1) return 0;
        if(minutesToDie>=minutesToTest)
        return buckets;
        double base = floor(minutesToTest/minutesToDie);
        double nb = buckets;
        int ans = 0;
        while(nb/base>=1){
            nb = ceil(nb/base);
            ans = ans+1;
        }
        return ans;
    }
};
```

## 485. Max Consecutive Ones

Given a binary array, find the maximum number of consecutive 1s in this array.

Example 1:
```
Input: [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s.
    The maximum number of consecutive 1s is 3.
```
**Note:**

The input array will only contain 0 and 1.
The length of input array is a positive integer and will not exceed 10,000

```cpp
class Solution {
public:
int findMaxConsecutiveOnes(vector<int>& nums) {
    int max=0,cur=0;
    for(int i=0;i<nums.size();i++)
    {
        if(nums[i]&1){
            max=max>++cur?max:cur;
        }
        else cur=0;
        
    }
    return max;
    }
};
```





## 210. Course Schedule II

There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

For example:
```
2, [[1,0]]
```
There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1]
```
4, [[1,0],[2,0],[3,1],[3,2]]
```
There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. So one correct course order is [0,1,2,3]. Another correct ordering is[0,2,1,3].

Note:
The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.
```cpp
template<typename T>  
void Print(T* vec, int size)  
{  
    for(int i=0;i<size;i++){
        cout << vec[i] << " ";
    }  
    cout << endl;
}  


class Graph
{
    int V;    // No. of vertices
    list<int> *adj;    // Pointer to an array containing adjacency lists
    bool isCyclicUtil(int v, bool visited[], bool *rs);  // used by isCyclic()
    void topSortUtil(int v, bool visited[], bool *rs, stack<int>& order);  // used by isCyclic()
public:
    Graph(int V);   // Constructor
    void addEdge(int v, int w);   // to add an edge to graph
    vector<int> topSort();    // returns true if there is a cycle in this graph
    bool isCyclic();
};
 
Graph::Graph(int V)
{
    this->V = V;
    adj = new list<int>[V];
}
 
void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
}
 
// This function is a variation of DFSUytil() in http://www.geeksforgeeks.org/archives/18212
void Graph::topSortUtil(int v, bool visited[], bool *recStack,stack<int>& order)
{
    if(visited[v] == false)
    {
        // Mark the current node as visited and part of recursion stack
        visited[v] = true;
        recStack[v] = true;
 
        // Recur for all the vertices adjacent to this vertex
        list<int>::iterator i;
        for(i = adj[v].begin(); i != adj[v].end(); ++i)
        {
            
            topSortUtil(*i, visited, recStack, order);
        }
        order.push(v);
 
    }
    recStack[v] = false;  // remove the vertex from recursion stack
}


// This function is a variation of DFSUytil() in http://www.geeksforgeeks.org/archives/18212
bool Graph::isCyclicUtil(int v, bool visited[], bool *recStack)
{
    if(visited[v] == false)
    {

        // Mark the current node as visited and part of recursion stack
        visited[v] = true;
        recStack[v] = true;
 
        // Recur for all the vertices adjacent to this vertex
        list<int>::iterator i;
        for(i = adj[v].begin(); i != adj[v].end(); ++i)
        {
            if ( !visited[*i] && isCyclicUtil(*i, visited, recStack) )
                return true;
            else if (recStack[*i])
                return true;
        }
 
    }
    recStack[v] = false;  // remove the vertex from recursion stack
    return false;
}
 
// Returns true if the graph contains a cycle, else false.
// This function is a variation of DFS() in http://www.geeksforgeeks.org/archives/18212
bool Graph::isCyclic()
{
    // Mark all the vertices as not visited and not part of recursion
    // stack
    bool *visited = new bool[V];
    bool *recStack = new bool[V];
    for(int i = 0; i < V; i++)
    {
        visited[i] = false;
        recStack[i] = false;
    }
 
    // Call the recursive helper function to detect cycle in different
    // DFS trees
    for(int i = 0; i < V; i++)
        if (isCyclicUtil(i, visited, recStack))
            return true;
 
    return false;
}
 
// Returns true if the graph contains a cycle, else false.
// This function is a variation of DFS() in http://www.geeksforgeeks.org/archives/18212
vector<int> Graph::topSort()
{
    // Mark all the vertices as not visited and not part of recursion
    // stack
    bool *visited = new bool[V];
    bool *recStack = new bool[V];
    for(int i = 0; i < V; i++)
    {
        visited[i] = false;
        recStack[i] = false;
    }
 
    // Call the recursive helper function to detect cycle in different
    // DFS trees
    stack<int> order;
    for(int i = 0; i < V; i++)
    topSortUtil(i, visited, recStack,order);
    vector<int> ts(this->V,0);
    for(int i=0;i<this->V;i++) {
        ts[i] = order.top();
        order.pop();
    }
    
    return ts;
}


class Solution {
public:
    vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites) {
        Graph G = Graph(numCourses);
        for(int i=0;i<prerequisites.size();i++){
            G.addEdge(prerequisites[i].second,prerequisites[i].first);
        }
        if(!G.isCyclic())
        return G.topSort();
        else return {}; 
    }
};
```


## 207. Course Schedule

There are a total of n courses you have to take, labeled from 0 to n - 1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?

For example:
```
2, [[1,0]]
```
There are a total of 2 courses to take. To take course 1 you should have finished course 0. So it is possible.

```
2, [[1,0],[0,1]]
```
There are a total of 2 courses to take. To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.

**Note:**
The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.

```cpp
class Graph
{
    int V;    // No. of vertices
    list<int> *adj;    // Pointer to an array containing adjacency lists
    bool isCyclicUtil(int v, bool visited[], bool *rs);  // used by isCyclic()
public:
    Graph(int V);   // Constructor
    void addEdge(int v, int w);   // to add an edge to graph
    bool isCyclic();    // returns true if there is a cycle in this graph
};
 
Graph::Graph(int V)
{
    this->V = V;
    adj = new list<int>[V];
}
 
void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
}
 
// This function is a variation of DFSUytil() in http://www.geeksforgeeks.org/archives/18212
bool Graph::isCyclicUtil(int v, bool visited[], bool *recStack)
{
    if(visited[v] == false)
    {
        // Mark the current node as visited and part of recursion stack
        visited[v] = true;
        recStack[v] = true;
 
        // Recur for all the vertices adjacent to this vertex
        list<int>::iterator i;
        for(i = adj[v].begin(); i != adj[v].end(); ++i)
        {
            if ( !visited[*i] && isCyclicUtil(*i, visited, recStack) )
                return true;
            else if (recStack[*i])
                return true;
        }
 
    }
    recStack[v] = false;  // remove the vertex from recursion stack
    return false;
}
 
// Returns true if the graph contains a cycle, else false.
// This function is a variation of DFS() in http://www.geeksforgeeks.org/archives/18212
bool Graph::isCyclic()
{
    // Mark all the vertices as not visited and not part of recursion
    // stack
    bool *visited = new bool[V];
    bool *recStack = new bool[V];
    for(int i = 0; i < V; i++)
    {
        visited[i] = false;
        recStack[i] = false;
    }
 
    // Call the recursive helper function to detect cycle in different
    // DFS trees
    for(int i = 0; i < V; i++)
        if (isCyclicUtil(i, visited, recStack))
            return true;
 
    return false;
}

class Solution {
public:
    


    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        Graph G = Graph(numCourses);
        for(int i=0;i<prerequisites.size();i++){
            G.addEdge(prerequisites[i].first,prerequisites[i].second);
        }
        return !G.isCyclic();
    }
};
```

## 261. Graph Valid Tree

Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

For example:

Given n = 5 and edges = `[[0, 1], [0, 2], [0, 3], [1, 4]]`, return true.

Given n = 5 and edges = `[[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]`, return false.

**Note**: you can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

```cpp
class Solution {
public:
        void print_vec(vector<int> vec){
        for(int i=0;i<vec.size();i++){
            cout<< vec[i] << " ";
        }
        cout << endl;
    }
    
    int root(vector<int> id, int i){
        while (i != id[i]) 
        i = id[i];
        return i;
    }
    
    int root_pc(vector<int> id,int i){
        while (i != id[i]) {
            id[i] = id[id[i]];
            i = id[i];
        }
        return i;
    }

    
    bool find(vector<int> id, int i, int j){
        return root(id,i)==root(id,j);
    }
    
    void unite(vector<int>& id, vector<int> sz, int p, int q){
        int i = root(id,p);
        int j = root(id,q);
        if(sz[i]<sz[j]){
            id[i] = j; sz[j]+=sz[i];
        }
        else{
            id[j] = i; sz[i]+=sz[j];
        }    
    }
    
    bool validTree(int n, vector<pair<int, int>>& edges) {
        vector<int> id(n,0);
        vector<int> sz(n,1);
        iota(id.begin(), id.end(), 0);
        for(int i=0;i<edges.size();i++){
            
            if(root_pc(id,edges[i].first)!=root_pc(id,edges[i].second)){
                unite(id,sz,edges[i].first,edges[i].second);
            }
            else{
                return false;
            }
            //print_vec(id);
        }
        unordered_set<int> cc;
        for(int i=0;i<n;i++){
            cc.insert(root(id,id[i]));
        }
        if(cc.size()!=1){
            return false;
        }
        else return true;
    }
};
```


## 296. Best Meeting Point

A group of two or more people wants to meet and minimize the total travel distance. You are given a 2D grid of values 0 or 1, where each 1 marks the home of someone in the group. The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.

For example, given three people living at (0,0), (0,4), and (2,2):
```
1 - 0 - 0 - 0 - 1
|   |   |   |   |
0 - 0 - 0 - 0 - 0
|   |   |   |   |
0 - 0 - 1 - 0 - 0
```
The point (0,2) is an ideal meeting point, as the total travel distance of 2+2+2=6 is minimal. So return 6.

```cpp
class Solution {

public:
    int minTotalDistance(vector<vector<int>>& grid) {
        vector<int> xvals;
        vector<int> yvals;
        for(int i=0;i<grid.size();i++){
            for(int j=0;j<grid[0].size();j++){
                if(grid[i][j]==1){
                    xvals.push_back(i);
                    yvals.push_back(j);
                }
            }
        }
        nth_element(xvals.begin(),xvals.begin()+xvals.size()/2,xvals.end());
        int xmid = xvals[xvals.size()/2];
        nth_element(yvals.begin(),yvals.begin()+yvals.size()/2,yvals.end());
        int ymid = yvals[yvals.size()/2];
        int mindist = 0;
        for(int i=0;i<xvals.size();i++){
            mindist += (abs(xvals[i]-xmid) + abs(yvals[i]-ymid));
        }
        return mindist;
    }
};
```


## 462. Minimum Moves to Equal Array Elements II
Given a non-empty integer array, find the minimum number of moves required to make all array elements equal, where a move is incrementing a selected element by 1 or decrementing a selected element by 1.

You may assume the array's length is at most 10,000.

**Example:**
```
Input:
[1,2,3]

Output:
2
```
Explanation:
Only two moves are needed (remember each move increments or decrements one element):

```
[1,2,3]  =>  [2,2,3]  =>  [2,2,2]
```

```cpp
class Solution {
public:

int findmedian(vector<int> nums){
    int median;
    if(nums.size()%2==0){ // even
        int n1 = kthsmallest(nums,nums.size()/2,0,nums.size()-1);
        int n2 = kthsmallest(nums,nums.size()/2+1,0,nums.size()-1);
        //cout << n1 << " "<< n2 << endl;
        return (n1+n2)/2;
    }
    else
    return kthsmallest(nums,nums.size()/2+1,0,nums.size()-1);
}

int kthsmallest(vector<int>& nums,int k, int start, int end){
    if(start == end) {
        //cout << "start = end =" <<start << " k = " << k << endl;
        return nums[start];
    }
    //cout << "array before partition>>>" << endl;
    //print_vec(nums);
    int pid = randpartition(nums, start, end);
    //cout << "array after partition>>>" << endl;
    //print_vec(nums);
    //cout << "pid = " << pid << endl;
    if(pid+1 == k) {
        return nums[pid];
    }
    else if(k<pid+1){
        //cout << " call left::: " << start << " "<< pid-1<<endl;
        return kthsmallest(nums,k,start,pid-1);    
    }
    else{
        //cout << " call right::: " << pid+1 << " "<< end<<endl;
        return kthsmallest(nums,k,pid+1,end);
    }
}

int randpartition(vector<int>& nums,int b, int e){
    //cout << " partitioning..." << b << " " << e<< endl;
    int p = b+(rand()%(e-b+1)); // randomized pivot
    //cout << "pivot id = " << p << endl;
    swap(nums[e],nums[p]);
    int pid = 0;
    //cout << "pivot = " << nums[e] << endl;
    for(int i=0;i<e;i++){
        if(nums[i]<=nums[e]){
            swap(nums[i],nums[pid]);
            pid++;
        }
    }
    swap(nums[pid],nums[e]);
    return pid;
}
    void print_vec(vector<int> nums){
        for(int i=0;i<nums.size();i++){
            cout << nums[i] << " ";
        }
        cout << endl;
    }
    int minMoves2(vector<int>& nums) {
        int n = findmedian(nums);
        //cout << "median =" << n << endl;
        int nb_moves = 0;
        for(int i=0;i<nums.size();i++){
            nb_moves+=abs(n-nums[i]);
        }
        return nb_moves;
    }
};

```

## 26. Remove Duplicates from Sorted Array

Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

For example,
Given input array nums = `[1,1,2]`,

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the new length.

```cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int j=0;
        int i=0;
        while(j<nums.size()){
            vector<int>::iterator ub;
            ub = upper_bound(nums.begin()+j+1,nums.end(),nums[j]);
            j = ub-nums.begin();
            nums[i++] = nums[j-1];
        }
        return i;
    }
};
```

## 27. Remove Element

Given an array and a value, remove all instances of that value in place and return the new length.

Do not allocate extra space for another array, you must do this in place with constant memory.

The order of elements can be changed. It doesn't matter what you leave beyond the new length.

Example:
```
Given input array nums = [3,2,2,3], val = 3
Your function should return length = 2, with the first two elements of nums being 2.
```

```cpp
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int i=0;
        for(int j=0;j<nums.size();j++){
            if(nums[j]!=val){
                nums[i++]=nums[j];
            }
        }
        return i;
    }
};
```

## 283. Move Zeroes

Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].

Note:

 1. You must do this in-place without making a copy of the array.
 2. Minimize the total number of operations.

```cpp
bool compare0(int x, int y)
{
  if(x==0) return false;
  else if(y==0) return true;
  else return x<y;
}
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int i=0;
        for(int j=0;j<nums.size();j++){
            if(nums[j]!=0){
                nums[i++] = nums[j];
            }
        }
        for(;i<nums.size();i++){
            nums[i]=0;
        }
    }
};
```

## 7. Reverse Integer

Reverse digits of an integer.

Example1: x = 123, return 321
Example2: x = -123, return -321

click to show spoilers.

**Note:**
The input is assumed to be a 32-bit signed integer. Your function should return 0 when the reversed integer overflows.

```cpp
class Solution {
public:
    int reverse(int x) {
        long long int y=0;
        int sign = x<0?-1:1;
        long long int z=x;
        z= (sign==-1)? -z:z;
        cout << x << endl;
        vector<int> digits;
        long long int b=1;
        while(x/b!=0){
            int rem = z%10;
            z = (z-rem)/10;
            digits.push_back(rem);
            b=b*10;
            //cout << z << " " << rem << " " << b << endl;
        }
        b=1;
        for(int i=digits.size()-1;i>=0;i--){
            y+=digits[i]*b;
            if(y>INT_MAX || y< INT_MIN) return 0;
            b=b*10;
        }
        y=sign<0?sign*y:y;
        return y;
    }
};
```

## 6. ZigZag Conversion

The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
```
P   A   H   N
A P L S I I G
Y   I   R
```
And then read line by line: "PAHNAPLSIIGYIR"
Write the code that will take a string and make this conversion given a number of rows:
```
string convert(string text, int nRows);
```
convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR".

```cpp
class Solution {
public:
    string convert(string s, int numRows) {
        if(numRows==1) return s;
        vector<string> res(numRows,"");
        int inc = 1;
        int row = 0;
        int i=0;
        while(i<s.size()){
        res[row] = res[row].append(1,s[i]);
         if(i%(2*numRows-2)==0){
             inc = 1;
         }
         else if((i-numRows+1)%(2*numRows-2)==0){
             inc = 0;
         }
         row+= inc==1 ? 1:-1; 
         //cout << "row = " << row << endl; 
         i++;
        //for(int j=0;j<numRows;j++){
        //    cout << res[j] << endl;
        //}
        //cout<< "--------------------"<<endl;
        }
        string rs = "";
        for( i=0;i<numRows;i++){
            rs+=res[i];
        }
        return rs;
    }
};
```


## 67. Add Binary

Given two binary strings, return their sum (also a binary string).

For example,
a = "11"
b = "1"
Return "100".

```cpp
class Solution
{
public:
    string addBinary(string a, string b)
    {
        string s = "";
        
        int c = 0, i = a.size() - 1, j = b.size() - 1;
        while(i >= 0 || j >= 0 || c == 1)
        {
            c += i >= 0 ? a[i --] - '0' : 0;
            c += j >= 0 ? b[j --] - '0' : 0;
            s = char(c % 2 + '0') + s;
            c /= 2;
        }
        
        return s;
    }
};
```

## 66. Plus One

Given a non-negative integer represented as a non-empty array of digits, plus one to the integer.

You may assume the integer do not contain any leading zero, except the number 0 itself.

The digits are stored such that the most significant digit is at the head of the list.

```cpp
class Solution {
public:
    vector<int> plusOne(vector<int>& digits) {
        int carry =1;
        for(int i =digits.size()-1;i>=0;i--){
            if(carry==0) break;
            else{
                digits[i]+=1;
                if(digits[i]==10) {
                    digits[i]=0;
                    carry=1;
                }
                else{
                    carry =0;
                    break;
                }
            }
        }
        if(carry == 1){
            vector<int> res(1,1);
            res.insert(res.end(),digits.begin(),digits.end());
            return res;
        }
        return digits;
    }
};
```

##469. Convex Polygon

Given a list of points that form a polygon when joined sequentially, find if this polygon is convex (Convex polygon definition).

Note:

There are at least 3 and at most 10,000 points.
Coordinates are in the range -10,000 to 10,000.
You may assume the polygon formed by given points is always a simple polygon (Simple polygon definition). In other words, we ensure that exactly two edges intersect at each vertex, and that edges otherwise don't intersect each other.
Example 1:
```
[[0,0],[0,1],[1,1],[1,0]]

Answer: True
Explanation: 
```
![enter image description here](https://leetcode.com/static/images/problemset/polygon_convex.png)

```
[[0,0],[0,10],[10,10],[10,0],[5,5]]

Answer: False

Explanation: 
```
![enter image description here](https://leetcode.com/static/images/problemset/polygon_not_convex.png)
```cpp

class Solution {
public:
    bool isConvex(vector<vector<int>>& p) {
      long n = p.size(), prev = 0, cur;
      for (int i = 0; i < n; ++i) {
        vector<vector<int>> A; // = {p[(i+1)%n]-p[i], p[(i+2)%n]-p[i]}
        for (int j = 1; j < 3; ++j) A.push_back({p[(i+j)%n][0]-p[i][0], p[(i+j)%n][1]-p[i][1]});
        if (cur = det2(A)) if (cur*prev < 0) return false; else prev = cur;
      }
      return true;
    }
    // calculate determinant of 2*2 matrix A
    long det2(vector<vector<int>>& A) { return A[0][0]*A[1][1] - A[0][1]*A[1][0]; }
```
/*
Note: Try to solve this task in linear time, since this is what you'll be asked to do during an interview.

Given a singly linked list consisting only of 0, 1, and 2, return it sorted in ascending order.

Example

For l = [2, 1, 0], the output should be
sortZeroOneTwoList(l) = [0, 1, 2];
For l = [0, 1, 0, 1, 2, 0], the output should be
sortZeroOneTwoList(l) = [0, 0, 0, 1, 1, 2].
Input/Output

[execution time limit] 0.5 seconds (cpp)

[input] linkedlist.integer l

A singly linked list of integers consisting only of 0, 1, and 2.

Guaranteed constraints:
0 ≤ list size ≤ 105,
0 ≤ element value ≤ 2.

[output] linkedlist.integer

Return l, sorted in ascending order.


*/


// Singly-linked lists are already defined with this interface:
// template<typename T>
// struct ListNode {
//   ListNode(const T &v) : value(v), next(nullptr) {}
//   T value;
//   ListNode *next;
// };
//


ListNode<int> * sortZeroOneTwoList(ListNode<int> * l) {
    /*
    Linear time. 
    Dutch partition with linked lists. 
    can we do 3 pointers ?
    */
    
    ListNode<int> * left  =  new ListNode(0);
    ListNode<int> * leftEnd  =  left;
    
    ListNode<int> * ones = new ListNode(0);
    ListNode<int> * onesEnd = ones; 
    
    
    ListNode<int> * right = new ListNode(0);
    ListNode<int> * rightEnd = right;
    

    ListNode<int> * mid = l;
    ListNode<int> * temp;
    
    while(mid != nullptr){ 
        int val = mid->value;         
        temp = mid->next;
        mid->next = nullptr;
        if(val == 0){
            leftEnd->next = mid; 
            leftEnd = leftEnd->next;
        } else if(val == 1) {
            onesEnd->next = mid; 
            onesEnd = onesEnd->next;
        } else if(val == 2) {
            rightEnd->next = mid;
            rightEnd = rightEnd->next; 
        }
        mid = temp; 
    }
    
    
    temp = left->next; 
       leftEnd->next = ones->next;
    onesEnd->next = right->next;
    while(temp != nullptr) {
        temp = temp->next;
    }
    return left->next;
}

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
    ListNode* reverseList(ListNode* head) {
        if(head == nullptr){
            return nullptr;
        }
        ListNode* prev = head;
        ListNode * after = prev->next;
        
        head->next = nullptr;
        
        while(after != nullptr){
           // if(after) {
            ListNode* temp = after->next;
           // }
            
            after->next = prev;
            
                
            prev = after;    
            after=temp;
            
        }
        
        return prev;
        
    }
};


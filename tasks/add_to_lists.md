## Add Two Numbers

[Leetcode](https://leetcode.com/problems/add-two-numbers/)  

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

#### Solution
```javascript
function ListNode(val, next) {
    this.val = (val===undefined ? 0 : val)
    this.next = (next===undefined ? null : next)
}
 
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */
var addTwoNumbers = function(l1, l2) {
    let l1c = l1;
    let l2c = l2;
    let res = null
    let cur = null
    let mem = 0;
    
    while (true) {
        let v;
        if (l1c && l2c) {
            v = l1c.val + l2c.val + mem;
        }
        else if (l1c) {
            v = l1c.val + mem;
        }
        else if (l2c) {
            v = l2c.val + mem;
        }
        else {
            if (mem == 0) break;
            v = mem;
        }
        mem = 0;
        if (v > 9) {
            v = v % 10;
            mem = 1;
        }
        
        n = new ListNode(v);
        if (cur) {
            cur.next = n;
        } else {
            res = n;
        }
        cur = n;
        
        
        if (l1c) l1c = l1c.next;
        if (l2c) l2c = l2c.next;
        if (!l1c && !l2c && mem == 0) {
            break;
        }
    }
    
    return res;
    
};
```

---
layout: single
title: "Stack and Queue Implementation with Doubly Linked List"
categories: algorithms
tags: [python,datastructures]
excerpt: "Python implementation of Doubly Linked List followed by Stack and Queue"
---

### Stack and Queue Implementation with Doubly Linked List
Lets start with defining `Node` and `DBList` classes.


```python
class Node:
    def __init__(self,el):
        self.e=el
        self.prev=None
        self.next=None
        
class DBList:
    def __init__(self):
        self.nil=Node(None)
        self.nil.next=self.nil
        self.nil.prev=self.nil
        
    def getListStr(self):
        nextnode=self.nil.next
        el=[]
        while nextnode!=self.nil:
            el.append(nextnode.e)
            nextnode=nextnode.next
        return str(el)
    
    def __repr__(self):
        return 'DBList: '+ self.getListStr()     
    
    def insert(self,el):
        z=Node(el)
        z.next=self.nil.next
        z.prev=self.nil
        self.nil.next.prev=z
        self.nil.next=z
        return self
    
    def delete(self,n):
        n.prev.next=n.next
        n.next.prev=n.prev
        return self
    
    def search(self,el):
        x=self.nil.next
        while x!=self.nil and x.e!=el:
            x=x.next
        return x
    
##Some tests
L=DBList()
L.insert(5)
L.insert(4)
L.insert(2)
print L
L.delete(L.search(2))
print L
L.delete(L.search(5))
print L
```

    DBList: [2, 4, 5]
    DBList: [4, 5]
    DBList: [4]


### Queue
Now it we have Doubly Linked List with insertion and deleteon in O(1) time. Note that use nil node to access last and first node in O(1) time. Lets define `Queue`. 


```python
class Queue:
    def __init__(self):
        self.q=DBList()
    def __repr__(self):
        return 'Queue: '+self.q.getListStr()
    def enque(self,e):
        self.q.insert(e)
    def peek(self):
        return self.q.nil.prev.e
    def deque(self):
        lastnode=self.q.nil.prev
        if lastnode==self.q.nil: return None
        self.q.delete(lastnode)
        return lastnode.e
q1=Queue()
q1.enque(5)
q1.enque(3)
print q1
print q1.peek()
print q1.deque()
print q1.deque()
print q1.deque()
q1.enque(42)
print q1.peek()
print q1

```

    Queue: [3, 5]
    5
    5
    3
    None
    42
    Queue: [42]


### Stack
Quite similar 


```python
class Stack:
    def __init__(self):
        self.q=DBList()
    def __repr__(self):
        return 'Stack: '+self.q.getListStr()
    def push(self,e):
        self.q.insert(e)
    def peek(self):
        return self.q.nil.next.e
    def pop(self):
        nextnode=self.q.nil.next
        if nextnode==self.q.nil: return None
        self.q.delete(nextnode)
        return nextnode.e
q1=Stack()
q1.push(5)
q1.push(3)
print q1
print q1.peek()
print q1.pop()
print q1.pop()
print q1.pop()
q1.push(42)
print q1.peek()

```

    Stack: [3, 5]
    3
    3
    5
    None
    42


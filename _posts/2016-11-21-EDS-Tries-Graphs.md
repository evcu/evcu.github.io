---
layout: single
title: "Trie and BFS & DFS implementation"
categories: algorithms
tags: [python,graph,datastructures]
excerpt: "I've implemented Trie and BFS/DFS with python"
---

### Tries for Prefix
This is my Trie implementation.
- Each node represents a char. 
- Two node for each Trie: Root(start of any word) and END(endofword flag for each word)
- `insertWord`, `hasPrefix`, `wordsWithPrefix` are the tree main functions


```python
class Node:
    def __init__(self,e):
        self.el=e
        self.children={}
    def __repr__(self):
        return str(self.el)+str(self.children)

class Trie:
    def __init__(self,vocabulary):
        self.root=Node('Root')
        self.endNode=Node('END')
        for word in vocabulary:
            self.insertWord(word)
    def __repr__(self):
        return str(self.root)
    def insertWord(self,w):
        nextnode=self.root
        for c in w:
            if c in nextnode.children: 
                nextnode=nextnode.children[c]
            else:
                newnode=Node(c)
                nextnode.children[c]=newnode
                nextnode=newnode
        nextnode.children['END']=self.endNode
    def hasPrefix(self,pref):
        nextnode=self.root
        for c in pref:
            if c in nextnode.children:
                nextnode=nextnode.children[c]
            else: return False
        return True
    def _discoverWords(self,node,pref):
        out=[]
        for child in node.children:
            if child=='END': out.append(pref)
            else: out.extend(self._discoverWords(node.children[child],pref+child))
        return out
    def wordsWithPrefix(self,pref):
        #returns list of words starting with pref
        nextnode=self.root
        for c in pref:
            if c in nextnode.children:
                nextnode=nextnode.children[c]
            else: return []
        return self._discoverWords(nextnode,pref)
```

Lets do some testing. 


```python
vocab=['ali','ata','bak','almak','balkon']
a=Trie(vocab)
print a
print a.hasPrefix('a')
print a.hasPrefix('at')
print a.hasPrefix('c')
print a.hasPrefix('bal')
print a.hasPrefix('bali')
print a.wordsWithPrefix('ba')

```

    Root{'a': a{'l': l{'i': i{'END': END{}}, 'm': m{'a': a{'k': k{'END': END{}}}}}, 't': t{'a': a{'END': END{}}}}, 'b': b{'a': a{'k': k{'END': END{}}, 'l': l{'k': k{'o': o{'n': n{'END': END{}}}}}}}}
    True
    True
    False
    True
    False
    ['bak', 'balkon']


### BFS and DFS implementation
I use a similar node class


```python
class Node:
    def __init__(self,e):
        self.el=e
        self.children={}
        self.mark=0
    def __repr__(self):
        return str(self.el)+str(self.children)
    def addChildren(self,nodes):
        for anothernode in nodes:
            self.children[anothernode.el]=anothernode
    def markit(self): self.mark=1
    def unmark(self): self.mark=0
        
from collections import deque
class Graph:
    def __init__(self,nodes):
        self.nodes={n.el:n for n in nodes}
    def unmarkGraph(self):
        for n in self.nodes.values():
            n.unmark()
    def DFS(self,startNode,fvisit):
        fvisit(startNode)
        startNode.markit()
        for n in startNode.children.values():
            if not n.mark:
                self.DFS(n,fvisit)
    def BFS(self,startNode,fvisit):
        visitq=deque()
        visitq.appendleft(startNode)
        startNode.markit()
        while visitq:
            nextnode=visitq.pop()
            fvisit(nextnode)
            for ch in nextnode.children.values():
                if not ch.mark: 
                    visitq.appendleft(ch)
                    ch.markit()
```

Lets define the following graph

![graph](/assets/images/graph.png)


```python
nl=[Node(0),Node(1),Node(2),Node(3),Node(4),Node(5)]
nl[0].addChildren([nl[5], nl[1],nl[2]])
nl[1].addChildren([nl[4],nl[3]])
nl[2].addChildren([nl[1]])
nl[3].addChildren([nl[4], nl[2]])
myg=Graph(nl)
def visitfun(x): print x.el,
myg.unmarkGraph()
print 'DFS->',
myg.DFS(nl[0],visitfun)
print
myg.unmarkGraph()
print 'BFS->',
myg.BFS(nl[0],visitfun)
print
```

    DFS-> 0 1 3 2 4 5
    BFS-> 0 1 2 5 3 4


---
layout: single
title: "Weekend Algorithm Problems Selection"
tags: [python,stack,graph,leetcode]
category: algorithms
excerpt: "Another busy weekend and Leet Code's Weekly Contest. Lets see three selectedd algorithm question with their solutions"
---

I hate this. It is just removed. I remember when I was dealing with writing a lot and then everything is lost. Luckily we have some git. It's saturday night and for some reason LeetCode schedules its weekly contest on Saturday. Whatif I had some life and wanna go out? That is not the case tonight. So lets start

### [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
I think a great problem and a really elegant solution with stack. Bascially keep your stack ordered and you need to pop, if you ended up seeing a value smaller than the current max. 


```python
def largestRectangleArea(heights):
    """
    :type heights: List[int]
    :rtype: int
    """
    stack = [(float('-inf'),None)]
    cmax = 0
    for i,e in enumerate(heights):
        if e > stack[-1][0]:
            stack.append((e,i))
        elif e < stack[-1][0]:
            while e < stack[-1][0]:
                tp = stack.pop()
                cmax = max(cmax,tp[0]*(i-tp[1]))
            stack.append((e,tp[1]))
    while len(stack) != 1:
        tp = stack.pop()
        cmax = max(cmax,tp[0]*(len(heights)-tp[1]))
    return cmax
```

Lets do some test! 


```python
print largestRectangleArea([2,13,2,3,0,18,9,23,3,3,2,3])
print largestRectangleArea([5,4,2,3,4,5,6])
```

    27
    14


### [200. Number of Islands](https://leetcode.com/problems/number-of-islands/)
We need to do a graph search and count the #connected components. To do that we can generate the graph, but we don't need a node abstraction each time. We already have the nodes as matrix abstraction

In other words we can just assume that the nodes with 1 are the nodes that are not visited yet and make a search on the whole graph. Number of DFS made are the #islands.

Instead of implementing the DFS-helper as a separete function, I've implemented a stack to do the Depth-First-Search.

Since there are $n*m$ vertices in the grid we aregoing to start theoretically at most that many dfs. However note that we are also setting the vertices zero once ve visited it. Therefore We not going to visit a vertice twice. A worst case scenario would be an all 1 graph. In this case we need to visit every vertice in the first call and then finish the for loop. This would cost $2n*m$ and therefore the solution is $O(n*m)$  


```python
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        n = len(grid)
        if n == 0: 
            return 0
        m = len(grid[0])
        if m == 0:
            return 0
        
        c = 0
        for i in xrange(n):
            for j in xrange(m):
                if grid[i][j] == '1':
                    #DFS with stack
                    c += 1
                    dfs_stack = [(i,j)]
                    while dfs_stack:
                        ci,cj = dfs_stack.pop()
                        if grid[ci][cj] == '1':
                            grid[ci][cj] = '0' 
                            if ci-1 >= 0:
                                dfs_stack.append((ci-1,cj))
                            if cj-1 >= 0:
                                dfs_stack.append((ci,cj-1))
                            if cj+1 < m:
                                dfs_stack.append((ci,cj+1))
                            if ci+1 < n:
                                dfs_stack.append((ci+1,cj))                    
        return c
```


```python
a = Solution()
a.numIslands([['1','0','1'],['0','0','0'],['1','0','1']])
```




    4



### [128. Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/)

The solution below is an O(n) solution. For each  non-elementary-interval [i,j] =[i,i+1,...,j-1,j] we hold two dictionary entry d[i]=j and d[j]=i. If the interval is elementary(single element) then we have only one dictionary element d[i] = i. Therefore the total entries in the dictionary is below n.

We pass the array once and every operation inside the loop is constant time. At each loop we are checking whether we can append the element(e) into an any interval. If so we make the necessary updates and conserve the interval properities like there is no


```python
class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ranges = {}
        for e in nums:
            if e in ranges:
                continue
            else:
                r_flag = (e+1) in ranges and ranges[e+1]>=(e+1)
                l_flag = (e-1) in ranges and ranges[e-1]<=(e-1)
                start = e
                end = e
                if l_flag:
                    start=ranges[e-1]
                    if start != ranges[start]:
                        del ranges[e-1]
                if r_flag:
                    end=ranges[e+1]   
                    if end != ranges[end]:
                        del ranges[e+1]
                ranges[start] = end
                ranges[end] = start    
        cmax = 0  
        for start,end in ranges.iteritems():
            cmax = max(cmax,end-start+1)
        return cmax
        
```

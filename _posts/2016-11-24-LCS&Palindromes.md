---
layout: single
title: "Palindromes and Longest Common Subsequence"
category: python teg2
excerpt: "Longest palindromes can be found with LCS on the string with its reverse"
---

### Longest Common Subsequence and Palindromes
A solution for finding Palindromes is using the Longest Common Subsequence(LCS) algorithm on the string and its reversed version. In this example I implement the LCS algorithm with dynamic programming and return all the longest common subsequences Here is an Dynamic Programming implementation of the Longest Common Subsequence algorithm.



```python
def findAllLCS(s1,s2):
    n = len(s1)
    m = len(s2)
    #For the length of longsest common substring
    c = [[0 for _ in range(m+1)] for _ in range(n+1)]
    #Path leading the longest length
    p = [[0 for _ in range(m)] for _ in range(n)] 
    for i in range(n):
        for j in range(m):
            if s1[i] == s2[j]:
                c[i+1][j+1] = c[i][j]+1
            elif c[i][j+1] > c[i+1][j]:
                c[i+1][j+1] = c[i][j+1]
                p[i][j] = 1
            elif c[i][j+1] < c[i+1][j]:
                c[i+1][j+1] = c[i+1][j]
                p[i][j] = -1
            else:
                c[i+1][j+1] = c[i+1][j]
                p[i][j] = 2
    allstrs=set()
    getLCS(s1,p,allstrs,n-1,m-1,'')
    return list(allstrs)
```


```python
def getLCS(s1,p,allstrs,n,m,cur_s):
    while n>=0 and m>=0:
        if p[n][m] == 0:
            cur_s=s1[n]+cur_s
            n -= 1
            m -= 1
        elif p[n][m] == 2:
            getLCS(s1,p,allstrs,n-1,m,cur_s)
            m -= 1
        elif p[n][m] == 1:
            n-=1
        else: #p[n][m] == -1
            m-=1
    allstrs.add(cur_s)
```

### Test Cases
An important note is, not all longest LCS's are palidrome. But there is at least one in the set of longest subsequences. 


```python
strs=['NYYNYNY','HEYHOE','NURSESRUN','XSSASDEQSASADASD']

for s in strs:
    print findAllLCS(s,s[::-1])
```

    ['NYYYN', 'NYNYY', 'YYNYY', 'YNYNY', 'YYNYN', 'NYNYN']
    ['EYE', 'HYE', 'EOE', 'HYH', 'EYH', 'EHE', 'HEH']
    ['NURSESRUN']
    ['SSASQSASS', 'SADSASDAS', 'SADASADAS', 'SSASESASS', 'SSASDSASS']


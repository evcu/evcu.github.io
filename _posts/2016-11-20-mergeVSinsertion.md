---
layout: single
title: "Merge vs Insertion Sort"
category: python
excerpt: "The algorithms class started with insertion sort and merge sort. Lets compare them"
---

### Merge Sort vs Insertion Sort
Lets define the useful swap


```python
def swap (a,i,j):
    t=a[i]
    a[i]=a[j]
    a[j]=t
```

### Insertion Sort


```python
def insort1(A,b):
    if b==1:
        return
    insort1(A,b-1)
    key=A[b-1]
    i=b-2
    while i>=0 and A[i]>key:
        A[i+1]=A[i]
        i-=1
    A[i+1]=key
```


```python
def insort2(A,b):
    if b==1:
        return
    insort2(A,b-1)
    for i in range(b-1,0,-1):
        if A[i]>A[i-1]:
            break
        else:
            swap(A,i,i-1)
```

### Merge Sort


```python
def mergee(a,s,m,e):
    n1 = m-s+1
    n2 = e-m
    a1 = a[s:(m+1)]
    a2 = a[(m+1):(e+1)]
    i = 0
    j = 0
    for k in xrange(s,e+1):
        if i>=n1:
            a[k:(e+1)] = a2[j:]
            break
        elif j>=n2:
            a[k:(e+1)] = a1[i:]
            break
        else:
            if a1[i]<a2[j]:
                a[k] = a1[i]
                i += 1
            else:
                a[k] = a2[j]
                j += 1
    
```


```python
def mesort(A,n):
    mesort_helper(A,0,n-1)

def mesort_helper(A,i,j):
    if i==j:
        return
    half=int((i+j)/2)
    mesort_helper(A,i,half)
    mesort_helper(A,half+1,j)
    mergee(A,i,half,j)
```

## Test Cases
I define my tester function and test cases


```python
import random
tests = [range(-10,11),
         range(-100,111,5),
         range(-10000,11111,40)]

def testIt(fun):
    n = len(tests)
    correct = n
    for test in tests:
        temp = test[:]
        random.shuffle(temp)
        fun(temp,len(temp))
        if temp != test:
            mistakes -= 1
    #print '%d/%d tests passed' % (correct,n)
    

```

Now I will time them and see the performance difference 


```python
%timeit testIt(insort1)

```

    100 loops, best of 3: 10.8 ms per loop



```python
%timeit testIt(insort2)


```

    10 loops, best of 3: 25.3 ms per loop



```python
%timeit testIt(mesort)

```

    100 loops, best of 3: 2.11 ms per loop


**A quick note:** An erray with 2000 element causes the following error for insertion sort: `Maximum recursion depth exceed`

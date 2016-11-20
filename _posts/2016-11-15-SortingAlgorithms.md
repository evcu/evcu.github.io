---
title: Denemece
category: python
excerpt: "Python implementation of some sorting"

---


```python
def test(f,*args):
    ts=([1,4,5,6,2,3,4,6,4,1,2,3,5,6,7,4],[1,2,3,4],[4,3,2,1],[1,1,1,1])
    c=1
    for singletest in ts:
        print 'Test'+str(c)+': '+','.join(str(s) for s in singletest)
        b=sorted(singletest)
        a=singletest[:]
        evalargs=[eval(ss) for ss in args]
        if len(args)==0: f(a)
        else: f(a,*evalargs)
        if a==b: print 'OK: '+','.join(str(s) for s in a)
        else:  print 'NO: Test'+str(c)+': '+ ','.join(str(s) for s in singletest)+'->'+','.join(str(s) for s in a)
        c+=1

def insertionsort(ls):
    l=len(ls)
    for i in range(1,l):
        ikey=ls[i]
        j=i-1
        while j>=0 and ls[j]>ikey:
            ls[j+1]=ls[j]
            j-=1
        ls[j+1]=ikey
    
test(insertionsort)
```

    Test1: 1,4,5,6,2,3,4,6,4,1,2,3,5,6,7,4
    OK: 1,1,2,2,3,3,4,4,4,4,5,5,6,6,6,7
    Test2: 1,2,3,4
    OK: 1,2,3,4
    Test3: 4,3,2,1
    OK: 1,2,3,4
    Test4: 1,1,1,1
    OK: 1,1,1,1



```python
def merg(ls,p,q,r):
    a=ls[p:(q+1)]
    a.append(float("inf"))
    b=ls[(q+1):(r+1)]
    b.append(float("inf"))
    i=j=0
    for k in range(p,(r+1)):
        if a[i]>b[j]:
            ls[k]=b[j]
            j+=1
        else:
            ls[k]=a[i]
            i+=1

def mergesort(ls,p,r):
    if r<=p: return
    q=(r+p)//2
    mergesort(ls,p,q)
    mergesort(ls,q+1,r)
    merg(ls,p,q,r)
    
test(mergesort,'0','len(a)-1')
```

    Test1: 1,4,5,6,2,3,4,6,4,1,2,3,5,6,7,4
    OK: 1,1,2,2,3,3,4,4,4,4,5,5,6,6,6,7
    Test2: 1,2,3,4
    OK: 1,2,3,4
    Test3: 4,3,2,1
    OK: 1,2,3,4
    Test4: 1,1,1,1
    OK: 1,1,1,1



```python

```

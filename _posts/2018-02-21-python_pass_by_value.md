---
layout: single
title: "Passing variables by value in Python "
tags: [python]
category: notes
excerpt: "Python closures are great, but what if you wanna pass the variable by value"
---

### Passing variables by value in Python

Let say you want to create a function that uses a variable from it's outer scope.
By default python creates a closure for each of these functions created and the variables
are evaluated during the call to the function. And you need to be careful about these
when you are coding!

```python
f_list=[]
for i in range(5):
    f_list.append(lambda a:print(a*i,end=','))

for f in f_list: f(5) #prints: 20,20,20,20,20,
```

In this example we are intended to create a list of functions that prints various multiple's
of input values. However since the variable `i` is passed by reference, it is not bounded until
we call the functions in the second loop. Since the `i` is set to be for at the end of the first
loop all of the functions multiply the input with 4.

So how can we evaluate the value of i during the definition and bound the variable `i` to
its current value. One way to do it is to use named arguments with default values! A

```python
f_list2=[]
for i in range(5):
    f_list2.append(lambda a,i=i: print(a*i,end=','))

for f in f_list2: f(5)  #prints: 0,5,10,15,20,

```

And it works like
a charm. Please use it on your `lambda` or regular `def` functions ;)

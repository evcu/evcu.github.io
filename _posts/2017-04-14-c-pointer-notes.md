---
layout: single
title: "C Pointers and Arrays Tutorial [Notes]"
tags: [tutorial,lang,c]
category: notes
excerpt: "notes from the pointer-land of C"
---

## C Pointers&Arrays Tutorial,
This is my notes from the 57 pages long [tutorial](https://pdos.csail.mit.edu/6.828/2014/readings/pointers.pdf) of C Pointers and Arrays.

### Dictionary.h
```c
int i; 
int *p;
int arr[10];
char *pc;
char arr[80];
struct st {
    char ss[10];
    int ii;
    float ff;
};
struct st my_st;
struct st *p_st;

int mx[ROW][COL];
typedef int RowArray[10];
RowArray *p_Ra; //array of RowArrays
```

I will use the variables throught out the notes
### Basics: Pointer points Variable
- `p = NULL` null pointer
- `*p=5; i=5;` value assignment
- `p=&i;` pointer assignment
- `printf("%p-%p", p, (void *)&i);` pointer printing
- `p+1; ++p; p++;` same and increments the pointer to the next addres according to the pointers type.

### Arrays are constant pointers
- `*(p+i)` === `arr[i]`
- `OK: p=arr` p points the array
- `NO: arr=p` since arr is a CONST! its address is determined at compile time!
- `void *pv` different type of pointers can't be compared/printed, so make them void!

### Strings are char arrays with '\0' ending.
- `puts(strA);` print the string to the stdout.
- String is a `*char`===`char[]` ending with `\0` char. 
- `strA[80]="asds"` this adds null char by default to the end.
- `char *strcpy(char *des, const char *source)` constant char to ensure it is not modified.
- `*p++`: returns the value, increments the pointer VS `(*p)++`: increments the value.
- If you want copy integers or other arrays, you need to provide N, too. There is no deliminator. 
- `a[3]`===`3[a]` haha! cumutative.
- `while(*pc)`===`while(*pc != '\0')` . '\0' is false.
- `char cc[] = "ted";`!==`char *cc = "ted";` array declaration adds the `\0` char!

### Structs are user defined variables and you point them!
- `fun(struct st *xx){}` what you
- `p_st=&my_st` make pointer to point the struct instance.
- `my_st.ii = 5` how to access fields...
- `(*p_st).ii = 8;` === `p_st->ii  = 8;` shortcut to get them

### Multi-Dim Arrays
- `*(*(mx + row) + col)` === `mx[row][col]` similar to the 1d case.
- `mx` === `&mx[0][0]`
- `mx[5][3]` __the main problem__ is one needs to know the length of the rows to skip 5 rows to access to the right one. `*(*(mx + row) + col)` in this example we need to increment mx row times, what row is now as big as `sizeof(int)*ROW`.
- `typedef int RowArray[10];` helps us at this point. It provides a fixed sized row data type.
- WAY1(Continious): `RowArray mx[5];` makes a matrix of 5*10. Now we know how much to skip. And one can do `RowArray *p = &mx[0];`, which is similar to `int *p=&arr[0];`.
- WAY2(''): `int (*p_Ra)[10];` === `RowArray *p_Ra;`, so here you define a matrix!!! And you can do double indexing and the pointer aritmetic correctly.
- WAY3(no): allocate pointers `int p_Ra=malloc(ROWS*sizeof(*int));` and then allocate rows inside a for loop `p_Ra[i] = malloc(COLS*sizeof(int));`.
- WAY4(yes): allocate pointers _as above_ and than do `p_Ra[0]=malloc(ROWS*COLS*sizeof(int));` and lastly do a for loop where you assign rest of the integer pointers `i=1:` and `p_Ra[i] = p_Ra[0] + (COLS*i)`. This is good, since we only make to malloc calls and __WAY3__ may get alot of malloc calls! and you need to `free(p_Ra[0]); free(p_Ra);`
### Dynamic Memory Allocation
- `p = (int *) malloc(10*sizeof(int));` +  `if (p == NULL) ERRROR` dynamic allocation
- `p_Ra = malloc(ROWS*sizeof(RowArray))` works I believe.


### Function stuff
- `void my_fun(struct st *p);`: prototype definition!, `struct st` is the name of the new data type: never `st` alone!
- `void sort(int a[],int N)` == `void sort(int *a,int N)`.
- `int (*func)(const void *p1);` this is pointer to a function!!!
- `so void sort(... , int (*f)(const void *))` thats how you write a function accepts a function and then you can call `sort(... , func);`!!!

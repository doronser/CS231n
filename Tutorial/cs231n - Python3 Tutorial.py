# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# #CS228 Python Tutorial
# %% [markdown]
# Adapted by [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) and [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335) from the `CS231n` Python tutorial by Justin Johnson (http://cs231n.github.io/python-numpy-tutorial/).
# %% [markdown]
# ##Introduction
# %% [markdown]
# Python is a great general-purpose programming language on its own, but with the help of a few popular libraries (numpy, scipy, matplotlib) it becomes a powerful environment for scientific computing.
# 
# We expect that many of you will have some experience with Python and numpy; for the rest of you, this section will serve as a quick crash course both on the Python programming language and on the use of Python for scientific computing.
# 
# Some of you may have previous knowledge in Matlab, in which case we also recommend the numpy for Matlab users page (https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html).
# %% [markdown]
# In this tutorial, we will cover:
# 
# * Basic Python: Basic data types (Containers, Lists, Dictionaries, Sets, Tuples), Functions, Classes
# * Numpy: Arrays, Array indexing, Datatypes, Array math, Broadcasting
# * Matplotlib: Plotting, Subplots, Images
# * IPython: Creating notebooks, Typical workflows
# %% [markdown]
# ##Basics of Python
# %% [markdown]
# Python is a high-level, dynamically typed multiparadigm programming language. Python code is often said to be almost like pseudocode, since it allows you to express very powerful ideas in very few lines of code while being very readable. As an example, here is an implementation of the classic quicksort algorithm in Python:

# %%
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]) )

# %% [markdown]
# ###Python versions
# %% [markdown]
# There are currently two different supported versions of Python, 2.7 and 3.4. Somewhat confusingly, Python 3.0 introduced many backwards-incompatible changes to the language, so code written for 2.7 may not work under 3.4 and vice versa. For this class all code will use Python 2.7.
# 
# You can check your Python version at the command line by running `python --version`.
# %% [markdown]
# ###Basic data types
# %% [markdown]
# ####Numbers
# %% [markdown]
# Integers and floats work as you would expect from other languages:

# %%
x = 3
print(x, type(x))


# %%
print(x + 1)   # Addition;
print(x - 1)   # Subtraction;
print(x * 2)   # Multiplication;
print(x ** 2)  # Exponentiation;


# %%
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"


# %%
y = 2.5
print(type(y)) # Prints "<type 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"

# %% [markdown]
# Note that unlike many languages, Python does not have unary increment (x++) or decrement (x--) operators.
# 
# Python also has built-in types for long integers and complex numbers; you can find all of the details in the [documentation](https://docs.python.org/2/library/stdtypes.html#numeric-types-int-float-long-complex).
# %% [markdown]
# ####Booleans
# %% [markdown]
# Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (`&&`, `||`, etc.):

# %%
t, f = True, False
print(type(t)) # Prints "<type 'bool'>"

# %% [markdown]
# Now we let's look at the operations:

# %%
print(t and f) # Logical AND;
print(t or f)  # Logical OR;
print(not t)   # Logical NOT;
print(t != f)  # Logical XOR;

# %% [markdown]
# ####Strings

# %%
hello = 'hello'   # String literals can use single quotes
world = "world"   # or double quotes; it does not matter.
print(hello, len(hello))


# %%
hw = hello + ' ' + world  # String concatenation
print(hw)                 # prints "hello world"


# %%
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)                             # prints "hello world 12"

# %% [markdown]
# String objects have a bunch of useful methods; for example:

# %%
s = "hello"
print(s.capitalize())           # Capitalize a string; prints "Hello"
print(s.upper())                # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))               # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))              # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())       # Strip leading and trailing whitespace; prints "world"

# %% [markdown]
# You can find a list of all string methods in the [documentation](https://docs.python.org/2/library/stdtypes.html#string-methods).
# %% [markdown]
# ###Containers
# %% [markdown]
# Python includes several built-in container types: lists, dictionaries, sets, and tuples.
# %% [markdown]
# ####Lists
# %% [markdown]
# A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:

# %%
xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
print(xs[-1])     # Negative indices count from the end of the list; prints "2"


# %%
xs[2] = 'foo'    # Lists can contain elements of different types
print(xs)


# %%
xs.append('bar') # Add a new element to the end of the list
print(xs)


# %%
x = xs.pop()     # Remove and return the last element of the list
print(x, xs)

# %% [markdown]
# As usual, you can find all the gory details about lists in the [documentation](https://docs.python.org/2/tutorial/datastructures.html#more-on-lists).
# %% [markdown]
# ####Slicing
# %% [markdown]
# In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing:

# %%
nums = list(range(5))    # range is a built-in function that creates a list of integers
print(nums)         # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])    # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print(nums[:-1])    # Slice indices can be negative; prints ["0, 1, 2, 3]"
nums[2:4] = [8, 9]  # Assign a new sublist to a slice
print (nums)        # Prints "[0, 1, 8, 9, 4]"

# %% [markdown]
# ####Loops
# %% [markdown]
# You can loop over the elements of a list like this:

# %%
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

# %% [markdown]
# If you want access to the index of each element within the body of a loop, use the built-in `enumerate` function:

# %%
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

# %% [markdown]
# ####List comprehensions:
# %% [markdown]
# When programming, frequently we want to transform one type of data into another. As a simple example, consider the following code that computes square numbers:

# %%
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)

# %% [markdown]
# You can make this code simpler using a list comprehension:

# %%
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)

# %% [markdown]
# List comprehensions can also contain conditions:

# %%
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)

# %% [markdown]
# ####Dictionaries
# %% [markdown]
# A dictionary stores (key, value) pairs, similar to a `Map` in Java or an object in Javascript. You can use it like this:

# %%
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"


# %%
d['fish'] = 'wet'    # Set an entry in a dictionary
print(d['fish'])     # Prints "wet"


# %%
print(d['monkey'])  # KeyError: 'monkey' not a key of d


# %%
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"


# %%
del d['fish']               # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"

# %% [markdown]
# You can find all you need to know about dictionaries in the [documentation](https://docs.python.org/2/library/stdtypes.html#dict).
# %% [markdown]
# It is easy to iterate over the keys in a dictionary:

# %%
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))

# %% [markdown]
# If you want access to keys and their corresponding values, use the iteritems method:

# %%
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))

# %% [markdown]
# Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries. For example:

# %%
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)

# %% [markdown]
# ####Sets
# %% [markdown]
# A set is an unordered collection of distinct elements. As a simple example, consider the following:

# %%
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"


# %%
animals.add('fish')      # Add an element to a set
print('fish' in animals)
print(len(animals))       # Number of elements in a set;


# %%
animals.add('cat')       # Adding an element that is already in the set does nothing
print(len(animals))       
animals.remove('cat')    # Remove an element from a set
print(len(animals))      

# %% [markdown]
# _Loops_: Iterating over a set has the same syntax as iterating over a list; however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:

# %%
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: dog", "#2: fish", "#3: cat"

# %% [markdown]
# Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:

# %%
from math import sqrt
print({int(sqrt(x)) for x in range(30)})

# %% [markdown]
# ####Tuples
# %% [markdown]
# A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:

# %%
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print(type(t))
print(d[t])       
print(d[(1, 2)])


# %%
t[0] = 1

# %% [markdown]
# ###Functions
# %% [markdown]
# Python functions are defined using the `def` keyword. For example:

# %%
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

# %% [markdown]
# We will often define functions to take optional keyword arguments, like this:

# %%
def hello(name, loud=False):
    if loud:
        print('HELLO, %s' % name.upper())
    else:
        print('Hello, %s!' % name)

hello('Bob')
hello('Fred', loud=True)

# %% [markdown]
# ###Classes
# %% [markdown]
# The syntax for defining classes in Python is straightforward:

# %%
class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"

# %% [markdown]
# ##Numpy
# %% [markdown]
# Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. If you are already familiar with MATLAB, you might find this [tutorial](http://wiki.scipy.org/NumPy_for_Matlab_Users) useful to get started with Numpy.
# %% [markdown]
# To use Numpy, we first need to import the `numpy` package:

# %%
import numpy as np

# %% [markdown]
# ###Arrays
# %% [markdown]
# A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.
# %% [markdown]
# We can initialize numpy arrays from nested Python lists, and access elements using square brackets:

# %%
a = np.array([1, 2, 3])  # Create a rank 1 array
print (type(a), a.shape, a[0], a[1], a[2])
a[0] = 5                 # Change an element of the array
print(a)                  


# %%
b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print(b)


# %%
print(b.shape)                   
print(b[0, 0], b[0, 1], b[1, 0])

# %% [markdown]
# Numpy also provides many functions to create arrays:

# %%
a = np.zeros((2,2))  # Create an array of all zeros
print(a)


# %%
b = np.ones((1,2))   # Create an array of all ones
print(b)


# %%
c = np.full((2,2), 7) # Create a constant array
print(c)


# %%
d = np.eye(2)        # Create a 2x2 identity matrix
print(d)


# %%
e = np.random.random((2,2)) # Create an array filled with random values
print(e)

# %% [markdown]
# ###Array indexing
# %% [markdown]
# Numpy offers several ways to index into arrays.
# %% [markdown]
# Slicing: Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:

# %%
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print(b)

# %% [markdown]
# A slice of an array is a view into the same data, so modifying it will modify the original array.

# %%
print(a[0, 1])
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1]) 

# %% [markdown]
# You can also mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the original array. Note that this is quite different from the way that MATLAB handles array slicing:

# %%
# Create the following rank 2 array with shape (3, 4)
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

# %% [markdown]
# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:

# %%
row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)


# %%
# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
col_r3 = a[:, [1]]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)
print(col_r3, col_r3.shape)

# %% [markdown]
# Integer array indexing: When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array. Here is an example:

# %%
a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and 
print(a[[0, 1, 2], [0, 1, 0]])

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))


# %%
# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))

# %% [markdown]
# One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:

# %%
# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)


# %%
# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"


# %%
# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10
print(a)

# %% [markdown]
# Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition. Here is an example:

# %%
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is > 2.

print(bool_idx)


# %%
# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])

# We can do all of the above in a single concise statement:
print(a[a > 2])

# %% [markdown]
# For brevity we have left out a lot of details about numpy array indexing; if you want to know more you should read the documentation.
# %% [markdown]
# ###Datatypes
# %% [markdown]
# Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype. Here is an example:

# %%
x = np.array([1, 2])  # Let numpy choose the datatype
y = np.array([1.0, 2.0])  # Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64)  # Force a particular datatype

print(x.dtype, y.dtype, z.dtype)

# %% [markdown]
# You can read all about numpy datatypes in the [documentation](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).
# %% [markdown]
# ###Array math
# %% [markdown]
# Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module:

# %%
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))


# %%
# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))


# %%
# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))


# %%
# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))


# %%
# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))

# %% [markdown]
# Note that unlike MATLAB, `*` is elementwise multiplication, not matrix multiplication. We instead use the dot function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:

# %%
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print (v.dot(w))
print (np.dot(v, w))


# %%
# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# %% [markdown]
# In order to avoide cumbersome code, the @ operator can be used for matrix multiplication: 

# %%
# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
print(x@y)

# %% [markdown]
# Numpy provides many useful functions for performing computations on arrays; one of the most useful is `sum`:

# %%
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"

# %% [markdown]
# You can find the full list of mathematical functions provided by numpy in the [documentation](http://docs.scipy.org/doc/numpy/reference/routines.math.html).
# 
# Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays. The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:

# %%
print(x)
print(x.T)


# %%
v = np.array([[1,2,3]])
print(v)
print(v.T)

# %% [markdown]
# ###Broadcasting
# %% [markdown]
# Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.
# 
# For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:

# %%
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)

# %% [markdown]
# This works; however when the matrix `x` is very large, computing an explicit loop in Python could be slow. Note that adding the vector v to each row of the matrix `x` is equivalent to forming a matrix `vv` by stacking multiple copies of `v` vertically, then performing elementwise summation of `x` and `vv`. We could implement this approach like this:

# %%
vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print(vv)                # Prints "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"


# %%
y = x + vv  # Add x and vv elementwise
print(y)

# %% [markdown]
# Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. Consider this version, using broadcasting:

# %%
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)

# %% [markdown]
# The line `y = x + v` works even though `x` has shape `(4, 3)` and `v` has shape `(3,)` due to broadcasting; this line works as if v actually had shape `(4, 3)`, where each row was a copy of `v`, and the sum was performed elementwise.
# 
# Broadcasting two arrays together follows these rules:
# 
# 1. If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
# 2. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
# 3. The arrays can be broadcast together if they are compatible in all dimensions.
# 4. After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
# 5. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension
# 
# If this explanation does not make sense, try reading the explanation from the [documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html) or this [explanation](http://wiki.scipy.org/EricsBroadcastingDoc).
# 
# Functions that support broadcasting are known as universal functions. You can find the list of all universal functions in the [documentation](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs).
# 
# Here are some applications of broadcasting:

# %%
# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:

print(np.reshape(v, (3, 1)) * w)


# %%
# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:

print(x + v)


# %%
# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:

print((x.T + w).T)


# %%
# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1) ) )


# %%
# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
print(x * 2)

# %% [markdown]
# Broadcasting typically makes your code more concise and faster, so you should strive to use it where possible.
# %% [markdown]
# This brief overview has touched on many of the important things that you need to know about numpy, but is far from complete. Check out the [numpy reference](http://docs.scipy.org/doc/numpy/reference/) to find out much more about numpy.

# %%
##SciPy

# %% [markdown]
# Numpy provides a high-performance multidimensional array and basic tools to compute with and manipulate these arrays. SciPy builds on this, and provides a large number of functions that operate on numpy arrays and are useful for different types of scientific and engineering applications.
# 
# The best way to get familiar with SciPy is to browse the documentation. We will highlight some parts of SciPy that you might find useful for this class.
# %% [markdown]
# ###Image operations
# %% [markdown]
# SciPy provides some basic functions to work with images. For example, it has functions to read images from disk into numpy arrays, to write numpy arrays to disk as images, and to resize images. Here is a simple example that showcases these functions:

# %%
from imageio import imread, imsave # Scipy.imread    is deprecated, use imageio instead
from skimage.transform import resize #Scipy.imresize is deprecated, use skimage.transform instead

# Read an JPEG image into a numpy array
img = imread('cat.jpg')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.5, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = resize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('cat_tinted.jpg', np.uint8(img_tinted))

# %% [markdown]
# ###Distance between points
# %% [markdown]
# SciPy defines some useful functions for computing distances between sets of points.
# 
# The function scipy.spatial.distance.pdist computes the distance between all pairs of points in a given set:

# %%
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
d = squareform(pdist(x, 'euclidean'))
print(d)

# %% [markdown]
# ##Matplotlib
# %% [markdown]
# Matplotlib is a plotting library. In this section give a brief introduction to the `matplotlib.pyplot` module, which provides a plotting system similar to that of MATLAB.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# By running this special iPython command, we will be displaying plots inline:

# %%
get_ipython().magic('matplotlib inline')

# %% [markdown]
# ###Plotting
# %% [markdown]
# The most important function in `matplotlib` is plot, which allows you to plot 2D data. Here is a simple example:
# %% [markdown]
# With just a little bit of extra work we can easily plot multiple lines at once, and add a title, legend, and axis labels:

# %%
# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)


# %%
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])

# %% [markdown]
# ###Subplots 
# %% [markdown]
# You can plot different things in the same figure using the subplot function. Here is an example:

# %%
# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()

# %% [markdown]
# You can read much more about the `subplot` function in the [documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot).
# %% [markdown]
# You can use the imshow function to show images. Here is an example:

# %%
import numpy as np
from scipy.misc import imresize
from imageio import imread # Scipy.imread is deprecated, use imageio instead
import matplotlib.pyplot as plt

img = imread('cat.jpg')
img_tinted = img * [1, 0.5, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()


# %%



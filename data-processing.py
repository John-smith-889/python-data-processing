# -*- coding: utf-8 -*-
"""

Agenda:

1. Operations over data structures
2. Control structures
3. OOP
4. Common excercises


1.General operations:
_______________________________________________________________________________

*  Insertion: addition of a new data element in a data structure. 
   Creating a new data structure I count in this category.
    
*  Deletion: Removal of a data element from a data structure if 
   it is found.

*  Merging: Combining elements of two data structures to create a new 
    data structure object.

_______________________________________________________________________________

*  Searching: searching for a data element with desired properties in 
   a data structure.
   I count Indexing to this category.

_______________________________________________________________________________

*  Traversal: processing all the data elements present in it.
   May be combined with searching if traversal is done under condition.
    
_______________________________________________________________________________

*  Sorting: Arranging data elements of a data structure in a specified order.
   I count reshaping in this category (if possible).
"""

###############################################################################
# List #
#!######

"""
Another data structures: Stacks and queues may be also implemented in Python 
lists.
"""

###############################################################
# Insertion #
#!###########


##################### 
# Create empty list #
##################### 

list_01 = []


########################################## 
# Generating lists with range() function #
##########################################

# Create list with numbers from 0 to 9
list_02 = list(range(10))
print(list_02)

# Create list with numbers from 1 to 9
list_03 = list(range(1, 10))
print(list_03)

# Create list with numbers from 1 to 9, by 2
list_04 = list(range(1, 10, 2))
print(list_04)


################################################# 
# Generating lists with arange() numpy function #
#################################################
# works exactly like range() but also support floats

import numpy as np 

list_05 = list(np.arange(10,20,0.1))
print(list_05)

list_06 = list(np.arange(20,10,-0.1))
print(list_06)



###############################################################
# Deletion #
#!##########

list_07 = list(range(10))

# remove element with a given index 
list_07.remove(list_07[0])

# remove element with a given value 
list_07.remove(9)

# remove from [0] to [1] elements from list
del list_07[0:2]
list_07


###############################################################
# Merging #
#!#########

#################
# merge 2 lists #
#################

list_03 + list_04


###############################################################
# Searching #
#!###########

list_08 = list(range(10))

##############
# Index list #
##############

list_08[1]

list_08[2:]

#######################################
# Search for elements under condition #
#######################################

[i for i in list_08 if i > 5] 


###############################################
# Search for elements indexes under condition #
###############################################

[i for i, value in enumerate(list_08) if value > 4] 
# enumerate() creating a list of tuples, where 1st elements of tuples
# are indexes of list elements

###########################################
# Check if certain element is in the list #
###########################################

10 in list_08

################################################
# Check how many given elements is in the list #
################################################

list_08.count(10)


###############################################################
# Traversal #
#!###########

import numpy as np
list_09 = np.arange(10,14,0.3333).tolist()
list_09

####################### 
# round list elements #
####################### 

list_10 = [round(i,2) for i in list_09] 
list_10

# or (round with map function to integers)
list(map(round,list_09))



#################################################
# create list of tuples/list with existing list # with zip function
#################################################

list_11 = list(zip(list_02,list_03))
list_11

#################################################
# create list of tuples/list with existing list # with list comprehension
#################################################

list_12 = [(i,2) for i in list_03]
list_12

# take only even elements 
list_13 = [(i,2) for i in list_03 if i % 2 == 0]
list_13

# Use nested list comprehension for generate tuples 
list_14 = [tuple( enum + p for p in list_04) for enum, i in enumerate(list_03)]
list_14

# create multiplication table
list_15 = list(range(1,11))
[[i*j for j in list_15] for i in list_15]




###############################################################
# Sorting #
#!#########

# sorting
sorted(list_04, reverse=False)

# sorting directly on object
list_04.sort(reverse=False)
list_04


###############################################
# Sorting with custom function (key argument) #
###############################################

# sorting string list with len function
sorted(['l','lala', 'la'], reverse=False, key=len)


# sort list of tuples by 2nd elements of tuples, with key argument
def f(element):
    return element[1]
# sort list with key
sorted([(1,2),(2,1),(3,4)], key=f)



# www.101computing.net/stacks-and-queues-using-python/
############################################################################### 
# Queue - FIFO - first in first out # 
#####################################
"""
First item which came to list will be the first to dequeue.
A queue is sometimes referred to as a First-In-First-Out (FIFO) 
or Last-In-Last-Out (LILO).
# https://stackoverflow.com/questions/10974922/what-is-the-basic-difference-between-stack-and-queue
"""

###############################################################
# Insertion #
#!###########

# Create queue
queue = [3, 4, 5]

# Add an element to the end of the queue
queue.append(6)
print(queue)


###############################################################
# Deletion #
#!##########

# remove (dequeue) element from the head of a queue
queue.remove(queue[0])


###############################################################################
# Stack - LIFO - last in first out # 
####################################
"""
The last element which was pushed on the stack will be the first to pop out.
LIFO or FILO - first in last out.
"""

###############################################################
# Insertion #
#!###########

# Create stack
stack = [3, 4, 5]
# append (push) 6 to the top of the stack
stack.append(6)
print(stack)


###############################################################
# Deletion #
#!##########

# pop the last element out of the stack
stack.pop() # optionally may put index in the brackets



###############################################################################
# Tuple #
#!#######

###############################################################
# Insertion #
#!###########
"""
Tuples in general are immutable in contradiction to lists.
"""

######################
# Create empty tuple #
######################

tuple_01 = ()
type(tuple_01)


tuple_02 = (1,2,3,4,5)


####################################
# Create tuple from range function #
####################################

tuple_03 = tuple(range(10))
tuple_03

############################ 
# Convert list_03 to tuple #
############################

tuple_04 = tuple(list_03)
tuple_04



###############################################################
# Deletion #
#!##########
"""
Tuples are immutable, but we still may filter it and create new tuples after
filtering.
"""

###############################################################
# Merging #
#!#########

################## 
# merge 2 tuples #
##################

tuple_03 + tuple_04


###############################################################
# Searching #
#!###########

################## 
# Indexing tuple #
##################

tuple_04[1]

tuple_04[:8]


###############################################################
# Traversal #
#!###########




###############################################################
# Sorting #
#!#########

tuple_05 = (1,2,5,3,5,2)
tuple_06 = tuple(sorted(tuple_05))
tuple_06

###############################################################################
# Dictionary #
#!############

###############################################################
# Insertion #
#!###########

##############################
# Create an empty dictionary #
##############################
dict_01 = {}
dict_01

# Create a dictionary with 3 key values, and 3 names
dict_02 = {1:"Adam",2:"Eve",3:"Steve"}
dict_02


################################ 
# Add to dictionary keys names #
################################

dict_02[4] = "Adam"
dict_02[5] = "Eve"
dict_02

#################################### 
# Create a dictionary from 2 lists #
####################################

dict_03 = dict(zip([1, 2], ['one', 'two']))
dict_03


###############################################################
# Deletion #
#!##########

# Delete element with given key
del dict_03[1]


###############################################################
# Merging #
#!#########

########################
# merge 2 dictionaries #
########################

# first one is base and we update first with this what new 2nd dict have 
dict_04 = {**dict_02, **dict_03}
dict_04

# or
dict_05 = dict_02.copy() # copy dict to new variable
dict_05
dict_05.update(dict_03)
dict_05


###############################################################
# Searching #
#!###########


####################### 
# Indexing dictionary #
#######################

dict_02[1]


#################################################
# Isolate keys and values of dictionary to list #
#################################################

list(dict_02.keys())

list(dict_02.values())


##########################################################
# Show dictionary as a list of tuples (easy for iterate) #
##########################################################

dict_02.items()


###############################################################
# Traversal #
#!###########


########################
# iterate a dictionary #
########################
# Iterate over a dictionary by converting the dictionary into list of tuples

for k,v in dict_02.items():
    print(k)


###############################################################
# Sorting #
#!#########

# Sort dictionary by value
dict_06 = dict(sorted(dict_02.items(), key=lambda x: x[1], reverse=True))

# or
def f(dict_):
    return dict_[1]
dict_07 = dict(sorted(dict_02.items(), key=f, reverse=True))


# Sort dictionary by key
dict_08 = dict(sorted(dict_02.items(), key=lambda x: x[0], reverse=False))



###############################################################################
# numpy array #
#!#############
# ndarray - n-dimensional array

###############################################################
# Insertion #
#!###########

################################
# Convert variables to ndarray #
################################

import numpy as np

# Convert list/tuple/set to ndarray (set first convert to list)
array_01 = np.array([1,2,3,4]).reshape(2,2)
type(array_01)

# Convert list of lists to ndarray 
array_02 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     [10,11,12],
                     [13,14,15]])
array_02


#######################################
# Create ndarray with arange() method #
#######################################

import numpy as np
array_03 = np.arange(1, 101, dtype = np.float).reshape(10,10)
array_03


#################################################
# Create ndarray using zeros() or ones() method #
#################################################

import numpy as np
array_04 = np.zeros((2, 3)) # array 2x3
array_04


######################################
# Create ndarray using random.rand() #
######################################
# (unifform [0,1) distribution)

array_05 = np.random.rand(5,3)  
array_05

################################
# Convert pandas df to ndarray #
################################

dataframe_01.values()

         
                
###############################################################
# Deletion #
#!##########


###############################################################
# Merging #
#!#########


###############################################################
# Searching #
#!###########


###############################################################
# Traversal #
#!###########


###############################################################
# Sorting #
#!#########





###############################################################################
# pandas dataframe #
#!##################

"""
In general pandas DataFrame is build on ndarray. For example, we may extract 
always ndarray from pandas dataframe using .values method.

And we may create pandas data frame with DataFrame() function (Capital letters
matters) using ndarray as 1st argument.   
"""

###############################################################
# Insertion #
#!###########

import pandas as pd

# create 5x3 pandas DataFrame from ndarray
dataframe_01 = pd.DataFrame(array_02,index=range(0,5,1),columns=list('ABC'))
                #
                #  index=range(0,5,1) - create index from 0, 5 poles, by 1
                #  columns=list('ABC')) - broke string to letters and add as          
type(dataframe_01)



###################################
# Create DataFrame from .csv file #
###################################

dataframe_02 = pd.read_csv("numbers3.csv")
dataframe_03 = pd.DataFrame(dataframe_02.values, index = (2,3),columns=['a','b','c'])
dataframe_03


###############################################################
# Deletion #
#!##########


###############################################################
# Merging #
#!#########


###############################################################
# Searching #
#!###########

######################
# DataFrame indexing #
######################

dataframe_03.iloc[3,0]


###############################################################
# Traversal #
#!###########


###############################################################
# Sorting #
#!#########





###############################################################################
# pandas series #
#!###############



###############################################################
# Insertion #
#!###########


###############################################################
# Deletion #
#!##########


###############################################################
# Merging #
#!#########


###############################################################
# Searching #
#!###########


###############################################################
# Traversal #
#!###########


###############################################################
# Sorting #
#!#########









###############################################################################
# Control structures #
######################



###############################################################
# for loop #
#!##########



###############################################################
# if, elif, else, break statements #
#!##################################



###############################################################
# while loop #
#!############

x = 100
while x > 0:
    print (x, end = " ")
    x = x - 10


###############################################################
# lambda function #
#!#################
# https://medium.com/swlh/lambda-vs-list-comprehension-6f0c0c3ea717


###############################################################
# list comprehension #
#!####################



###############################################################
# dictionary comprehension #
#!##########################



###############################################################
# map function #
#!##############
# https://www.geeksforgeeks.org/python-map-function/


###############################################################
# enumerate function #
######################





###############################################################################
# OOP #
#######

##################
# Create a class #
##################

class Printing_class_01:
    """
    Class for printing. 
    """

    # Constructor method
    def __init__(self, field_01, field_02):

        # Store value to field field_01 in the object
        self.field_01 = field_01
        self.field_02 = field_02 

    # print_01 method
    def print_01(self):
        """ Print  """
        print(self.field_01)
        print(self.field_02)
     
    # print_02 method   
    def print_02(self):
        """ Print double  """
        x = 2*(self.field_01)
        y = 2*(self.field_02)
        print(x)
        print(y)

    # print_03 method for passing computed parameters without printing
    def print_03(self):
        """ Printing parameters pass  """
        x = 2*(self.field_01)
        y = 2*(self.field_02)
        
        # returning variable x
        return x

# Create object  "printing_object_01"
printing_object_01 = Printing_class_01('la','lala')

# Run print_01 method
printing_object_01.print_01()

# Run print_02 method
printing_object_01.print_02()


###################################
# Create a class with inheritance #
###################################

class Printing_class_02(Printing_class_01):

    def print_03_boost(self):
        
        # Create variable with the longest lala
        longest_lala = 3*(self.print_03())
        
        # print the longest lala
        print(longest_lala)


# Create object  "printing_object_02"
printing_object_02 = Printing_class_02('la','lala')

# Run "print_02_boost" method
printing_object_02.print_03_boost()

# As we can see there is no need to define constructor, 
# because it is inherited from parent class



###############################################################################
# Common exercises #
####################


###############################################################
# factorial #
#############

def factorial(n):
    if n == 0: # 3 operations
        factorial = 1
        print(factorial)    
    elif n>0:                # ~ n+3 operations
        factorial=1
        for i in range(1,n+1):
            factorial=factorial*i    
        print(factorial) 
    elif n<0: # 2 operations
        print("n should be >= 0")

factorial(5)
# list(range(1,2+1))    



######################################################
# do operations on list/array until you encounter -1 #
######################################################

list_01 = [1,2,3,4,5,-1,2,3]

for i in list_01:
    if i == -1:
        break
    else:
        print(i) # operation






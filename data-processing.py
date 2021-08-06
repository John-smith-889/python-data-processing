# -*- coding: utf-8 -*-
"""

Scripts consists examples of operations over various data structures in Python:
-

Operations:
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
    
   In general to search (section above) in some data structure we need also 
   to traverse it or iterate over it. So Every search consists some
   form of traversal. In this section we focus on traversal 
   with not only searching but modificiation of data structire elements.
   
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

# Convert np array to list (with numpy integers to primitive int)
list_05 = np.arange(10,20,0.1).tolist()
print(list_05)

# Convert np array to list (still with numpy types)
list_06 = list(np.arange(20,10,-0.1))
print(list_06)

##########################
# insert element to list #
##########################

# Insert value 0 to position 1
list_04.insert(1, 0)

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


###################################################
# create list/tuple of tuples with existing lists # with zip function
###################################################

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



###############################################################
# Searching #
#!###########

list_08 = list(range(10))

##############
# Index list #
##############

list_08[1]

list_08[2:]


####################################
# How many elements is in the list #
####################################

len(list_08)


###########################################
# Check if certain element is in the list #
###########################################

3 in list_08

################################################
# Check how many given elements is in the list #
################################################

list_08.count(10)

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

#####################################
# Check mutual elements for 2 lists #
#####################################

[i for i in list_08 if i in [3,4]]

########################################### 
# Check if is at least one mutual element #
###########################################

any([i for i in list_08 if i in [3,4]])


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



#################################
# transform elements form lists # with list comprehension
#################################

# Use nested list comprehension for generate tuples 
list_14 = [tuple( enum + j for j in list_04) for enum, i in enumerate(list_03)]
list_14

###############################
# create multiplication table #
###############################

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

#######################################
# Search for elements under condition #
#######################################

tuple(i for i in tuple_02 if i % 2 == 0)


###############################################################
# Traversal #
#!###########

#########################
# modify tuple elements # with list comprehension
#########################

tuple(i for i in tuple_02 if i % 2 == 0)

###############################################################
# Sorting #
#!#########

tuple_05 = (1,2,5,3,5,2)
tuple_06 = tuple(sorted(tuple_05))
tuple_06



###############################################################################
# Set #
#!#####


###############################################################
# Insertion #
#!###########

#######################
# Create an empty set #
#######################
set_01 = set()
set_01
type(set_01) # set


##############
# Create set #
##############
set_02 = {0,1,3}
set_02
type(set_02) # set


###############################################################
# Searching #
#!###########

#=======================#
# Set - subset matching #
#=======================#

set_03 = {0,1,3}
set_04 = {0,1,3,4,5}
# Check if set_03 is a subset of set_04
set_03.issubset(set_04) # True



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

#####################################
# Searching with list comprehension #
#####################################

[v for k,v in dict_04.items() if k % 2 == 0]

###############################################################
# Traversal #
#!###########


########################
# iterate a dictionary #
########################
# Iterate over a dictionary by converting the dictionary into list of tuples

for k,v in dict_02.items():
    print(k)

# dict comprehension
{v:v for k,v in dict_03.items() if v % 2 == 0}


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

############################
# Convert lists to ndarray #
############################

import numpy as np

# Convert list/tuple/set to ndarray (set first convert to list)
array_01 = np.array([1,2,3,4,5]).reshape(5,1)
array_01

# Convert list of lists to ndarray 
array_02 = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9],
                     [10,11,12],
                     [13,14,15]])
array_02


#################################
# Insert data to columns / rows #
#################################
# possible to imput any numbers with indexing

array_02[0,:] = [1,2,5]
array_02
array_02[0,:] = 1
array_02


################################
# Convert pandas df to ndarray #
################################

dataframe_01.values


#######################################
# Create ndarray with arange() method #
#######################################
# Control over distance between generated elements

import numpy as np
array_03 = np.arange(1, 101, dtype = np.float).reshape(10,10)
array_03


#######################################
# Create ndarray with arange() method #
#######################################
# Control over number of generated elements

array_04 = np.linspace(1,11,2)


#################################################
# Create ndarray using zeros() or ones() method #
#################################################

import numpy as np
array_05 = np.zeros((2, 3)) # array 2x3
array_05


######################################
# Create ndarray using random.rand() #
######################################
# (unifform [0,1) distribution)

array_06 = np.random.rand(4,2)  
array_06



         
                
###############################################################
# Deletion #
#!##########

###########################
# Delete rows and columns #
###########################
# np.delete(arr, obj, axis = None)

# arr : Numpy array.
# obj : index position or list of index positions.
# axis : Axis along which we want to delete. If 1 then dlete columns, 
# if 0 then delete rows and if None then apply delete on flattened array.

del array_06 # delate whole array

np.delete(array_02, 0, 0) # delete 1 row
np.delete(array_02, 1, 0) # delete 2 row
np.delete(array_02, [0,1], 0) # delete list of rows

np.delete(array_02, 0, 1) # delete 1 column
np.delete(array_02, [0,1], 1) # delete list of column

np.delete(array_02, np.s_[0:1],0) # delete rows slice
#or
np.delete(array_02, slice(0,1),0) # delete rows slice
np.delete(array_02, np.s_[0:1],1) # delete columns slice


###############################################################
# Merging #
#!#########

array_06 = np.random.rand(4,2)  
array_06

# merge arrays (horizontal concatenation)
np.concatenate([array_06, array_06], axis=1)
np.concatenate([array_06, array_06, array_06], axis=1)

# merge arrays (vertical concatenation)
array_08 = np.arange(4).reshape(1,4)
# or 
array_08 = np.expand_dims(np.arange(4), axis=0) # (added one more dim)

np.concatenate((array_07, array_08), axis=0)



###############################################################
# Searching #
#!###########

import numpy as np

array_09 = np.random.randint(0,2,12).reshape(4,3)
array_09

############
# Indexing #
############
# intuitive indexing

array_09[:,0:2] # get columns slice
array_09[0] # get first row
array_09[[0,1],:] # get list of rows
array_09[[0],:] # square brackets keep sliced row/column 2-dimensional


# create array which consists of choosen multiple rows indexed by choosen column values
array_09[array_09[:,1]] 
# or
array_09[[1,0,0,1]]

################################
# Check which elements are NaN #
################################
array_10 = np.round(np.random.random(10).reshape(5,2), decimals = 2)
array_10[0,0] = 'nan' # nan cannot be imputed to integer type ndarray
array_10

# Return bool ndarray
np.isnan(array_10)

# Return ndarray with indices of NaN
np.argwhere(np.isnan(array_10))


###############################################################
# Traversal #
#!###########

#######################
# Iterate over column #
#######################

[i for i in array_10[:,0]]


####################
# Iterate over row #
####################

[i for i in array_10[0,:]]


########################
# Iterate over ndarray #
########################

# For loop
for (x,y),value in np.ndenumerate(array_02):
    print(x,y,value)

list(np.ndenumerate(array_02))

# List comprehension
[i for (x,y),i in np.ndenumerate(array_04) if x == 0 and y == 2 ]



###################################
# Convert all NaN values to zeros #
###################################

np.nan_to_num(array_10, copy=False)


#########
# Round #
#########

array_11 = np.round(np.random.random(10), decimals = 2)



###############################################################
# Sorting #
#!#########

array_12 = np.random.randint(0,2,12).reshape(3,4).T
array_12

#################################################
# Sort array according to sorted choosen column #
#################################################

array_12[array_12[:,1].argsort()]
# 1) take column - array_12[:,1]
# 2) return list with indexing column elements - array_12[:,1].argsort() 
# 3) apply those indices to whole ndarray as recombining row order


##############################################
# Sort array according to sorted choosen row #
##############################################

array_12[:,np.argsort(array_10[0,:])]


###################################
# Sort columns/rows independently #
###################################

# sort columns ascending (columns/rows sorted independently)
np.sort(array_06, axis=0) 
np.sort(array_06, axis=1)




###############################################################################
# pandas series #
#!###############

"""
Pandas series is a data structure similar to data frame with a few differences.
It has index and only one column.

Series is build on ndarray. We may extract 1 dimensional ndarray 
from pandas Series using .values method.

We may create pandas Series with Series() method (Capital letters
matters) using ndarray as 1st argument, or other data structures.   
"""

###############################################################
# Insertion #
#!###########

#===============================#
# Create sample timeseries data #
#===============================#
import numpy as np
import pandas as pd

# Empty Series
series_empty = pd.Series()

# Time indexed Series
datetime_index_01 = pd.date_range(start='2020-01-01', periods=10, freq='D') 
series_01 = pd.Series(np.arange(1,11),index=datetime_index_01)
series_01
type(series_01)

# Numbers indexed Series
series_02 = pd.Series([10, 20, 30], index=[0, 1, 2])
series_02

#===========================================#
# Convert pandas DataFrame to pandas Series #
#===========================================#

# Create DataFrame with single column
dataframe_01 = pd.DataFrame(np.arange(5),index=range(5), columns=['A'])

# Convert DataFrame to Series
dataframe_01.squeeze()
# name of a Series is automatically taken from column name


# Create DataFrame with multiple columns
dataframe_02 = pd.DataFrame(np.arange(10).reshape(5,2),index=range(5), columns=['A','B'])

# Convert specific column of DataFrame to Series
dataframe_02['A'].squeeze()

# Convert specific row of DataFrame to Series
dataframe_02.iloc[0].squeeze()
# columns names are taken automatically as index


#===========================================#
# Convert pandas Series to pandas DataFrame #
#===========================================#

series_02 = pd.Series([10, 20, 30], index=[0, 1, 2])
series_02

# Convers Series to DataFrame
series_02.to_frame()
# or
pd.DataFrame(series_02, columns=['A'])




###############################################################
# Deletion #
#!##########

# Delete Series object
del series_02

# get 9th and 10th element of DatetimeIndex
series_01.index[[8, 9]]

# Delete 2 elements from Series indexed by DatetimeIndex object
series_01.drop(series_01.index[[8, 9]])


###############################################################
# Merging #
#!#########

# Merge Series 
series_merged = series_01.append(series_01)
series_merged


###############################################################
# Searching #
#!###########

#===============================#
# Create sample timeseries data #
#===============================#

import numpy as np
import pandas as pd
datetime_index_01 = pd.date_range(start='2020-01-01', periods=10, freq='D') 
series_01 = pd.Series(np.arange(1,11),index=datetime_index_01)

# extract values of Series
series_01.values

# extract index of Series
series_01.index

#=====================#
# timeseries indexing #
#=====================#
#(taking 1st element of series)

from datetime import datetime
series_01.loc['2020-01-01']
series_01.loc[datetime(2020, 1, 1)]
series_01[0]
series_01[2:]
series_01[:-2]

#=====================================#
# taking the last DatetimeIndex value #
#=====================================#

# Extract last DatetimeIndex value (pandas timestamp) as timestamp
# (if properly sorted in ascending order)
series_01.index[-1]
# Extract second last DatetimeIndex value (pandas timestamp) as datetime
series_01.index[-1].to_pydatetime()
#or
series_01.last('1D').index[0].to_pydatetime()

#=====================================#
# Iteration and filtering over Series #
#=====================================#

# iterate over Series values 
for i in series_01:
    print(i)

# iterate over Series values under condition
for i in series_01:
    if i>5 and i<9: print(i)
    
# iterate over Series index and values 
for index, i in series_01.items():
    print(index, i)

# iterate over Series index and values under condition
from datetime import datetime
datetime_01 = datetime(year=2020, month=1, day=3)

for index, i in series_01.items():
    if index>datetime_01 and i>5: 
        print(index, i)

# filter Series
series_01.loc[(series_01 > 0) & (series_01 < 6)]


###############################################################
# Traversal #
#!###########

# Create sample timeseries data 
import numpy as np
import pandas as pd
datetime_index_01 = pd.date_range(start='2020-01-01', periods=10, freq='D') 
series_01 = pd.Series(np.arange(1,11),index=datetime_index_01)

# multiply every element of series times number
series_01 * 2

# perform operation on Series elements under condition
[i**2 for i in series_01 if i>5] 


###############################################################
# Sorting #
#!#########

# Create sample timeseries data 
import numpy as np
import pandas as pd
datetime_index_01 = pd.date_range(start='2020-01-01', periods=10, freq='D') 
series_01 = pd.Series(np.arange(1,11),index=datetime_index_01)

# sort values in ascending order
series_01.sort_values()

# sort values in descending order
series_01.sort_values(ascending=False)


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

#################################
# Create DataFrame from ndarray #
#################################

import numpy as np
import pandas as pd

dataframe_01 = pd.DataFrame(np.arange(1,21).reshape((5,4)),index=range(0,5,1), 
                            columns=list('ABCD'))
dataframe_01

# Add a new row to dataframe
new_row = {'A':9, 'B':9, 'C':9, 'D':9}
dataframe_01.append(new_row, ignore_index=True)


######################
# Convert dict to df #
######################
# each key - value pair is in one column

dict_01 = {'one': ['A', 'B', 'C', 'D'], 
              'two': [1, 2, 3, 4], 
              'three': ['a', 'b', 'c', 'd']}

dataframe_02 = pd.DataFrame(dict_01)


######################
# Convert list to df #
######################
# first element of each tuple is in one column

list_00 = [('one', 'A', 'B', 'C', 'D'), 
           ('two', 1, 2, 3, 4), 
           ('three', 'a', 'b', 'c', 'd')]

dataframe_02 = pd.DataFrame.from_records(list_00)


# TODO add from_dict method

###################################
# Create DataFrame from .csv file #
###################################

dataframe_03 = pd.read_csv("numbers3.csv", 
                           sep = ',', # also alias "delimiter"
                           header = None, # which row take as a column names
                           # None - names are generated from 0 up
                           names = list('abc'), # column names(overwrite header)
                           usecols = [0,1,2], # read particular columns,
                           # if no names, provide integer numbers of cols                           
                           skiprows = 0, # how many rows skip
                           nrows = 4, # number of rows we want to import
                           index_col = None, # set particular column as index
                           decimal = '.',
                           dtype={"a": int} ) #values: float, int, object
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
dataframe_03


# Change index and column names
dataframe_03 = pd.DataFrame(dataframe_03.values, index = [0,1,2], columns=['a','b','c'])
dataframe_03



###############################################################
# Deletion #
#!##########

###################
# Column deletion #
###################

del dataframe_03['a']

# or
dataframe_03.drop(['a','b'], axis=1, inplace=False)
dataframe_03.drop(dataframe_03.columns[[0, 1]], axis=1, inplace=False) 


#################
# Rows deletion #
#################

dataframe_03.drop([0,1], axis=0, inplace=False) # works if rows has proper 
                                                # indices '0', and '1'
dataframe_03.drop(range(2), axis=0, inplace=False)
# or
dataframe_03.drop(dataframe_03.index[[0,1]])
dataframe_03.drop(dataframe_03.index[[range(2)]])

###############################################################
# Merging #
#!#########

dataframe_04 = pd.DataFrame(np.arange(1,5).reshape(2,2))
dataframe_05 = pd.DataFrame(np.arange(5,9).reshape(2,2))

########################
# merge dfs vertically #
########################

pd.concat([dataframe_04, dataframe_05, dataframe_05], axis=0, sort=False)


##########################
# merge dfs horizontally #
##########################

pd.concat([dataframe_04, dataframe_05], axis=1, sort=False)


###########################
# Add column to pandas df #
###########################

dataframe_05[2] = ['one','two']
dataframe_05['A'] = [6,7]


column_01 = ['one','two']
dataframe_05[3] = column_01


###############################################################
# Searching #
#!###########

######################
# DataFrame indexing #
######################

dataframe_01[['A','B']] # get column as df
dataframe_01.A # get column as pd series
dataframe_01.A[0] # get particular value from column


######################################
# DataFrame slicing under conditions #
######################################
# filtering

dataframe_01[dataframe_01.A != 5]

dataframe_01[dataframe_01.A > 2][dataframe_01.B > 6]

# Take value of column B from row which meet condition of column A
dataframe_01[dataframe_01.A == 5]['B']

################################
# DataFrame indexing with iloc #
################################
# integer based location

# double, nested brackets enforce DataFrame output 
# single brackets enforce Series output

dataframe_01
dataframe_01.iloc[0,0] # get particular value

dataframe_01.iloc[[0]] # get one row
dataframe_01.iloc[0:2] # get slice of rows (2 first rows)
dataframe_01.iloc[[0,3]] # get particular rows

dataframe_01.iloc[[-1]] # get last row
dataframe_01.iloc[-2:] # get last 2 rows


dataframe_01.iloc[:,[0]] # get one column
dataframe_01.iloc[:,[0,1]] # get particular columns
dataframe_01.iloc[:,list(range(2))] # get particular columns

dataframe_01.iloc[0:2,[0]] # slice
dataframe_01.iloc[0:2,[0,1,2]] # slice
dataframe_01.iloc[[0,1],[0,1,2]] # slice


###############################
# DataFrame indexing with loc #
###############################
# label based location

dataframe_01
dataframe_01.loc[:,['A']] # get particular column
dataframe_01.loc[[0,1],['A']] # get particular rows and columns
dataframe_01.loc[0:2,['A','B']] # get rows slice and columns (involving row 2)


######################
# Check column types #
######################

dataframe_01.dtypes


##################
# query() method #
##################
# High performance memory-saving query

# Query df according to conditions
dataframe_01
dataframe_01.query('A>1 and B>3 and D in[8,12,16]', inplace = False)


####################
# pandasql quering #
####################
# Using pandasql module with SQLite syntax

import pandasql as ps

# Write queries
q1 = "SELECT * FROM dataframe_01"

q2 = """SELECT * 
        FROM dataframe_01
        WHERE C > 3     """

# Querying with sqldf() method
ps.sqldf(q1, globals())
ps.sqldf(q2, globals())


# Simplifying Queries with lambda function
sql = lambda query: ps.sqldf(query, globals())

# Querying with sqldf() method with lambda function
sql("select * from dataframe_01")



##################################
# Checking frequencies of values #
##################################
# value_counts() has to be performed on pd series

# Frequencies of choosen column
dataframe_01[0].value_counts()

# Frequencies of choosen row
dataframe_01.iloc[0].value_counts()


###############################################################
# Traversal #
#!###########

#######################
# Iterate over column #
#######################

[i**2 for i in dataframe_01.A]
[i**2 for i in dataframe_01[0]]

# assign list as a new A column
dataframe_01.A = [i**2 for i in dataframe_01.A]


####################
# Iterate over row #
####################
# using numpy array

# Iterate over choosen row
[i**2 for i in dataframe_01.values[0,:]]
# Iterate over choosen row and assign values
dataframe_01.iloc[[0],:] = [i**2 for i in dataframe_01.values[0,:]]


#########################
# Iterate over whole df #
#########################
# using ndarray

[i**2 for i in dataframe_01.values]
dataframe_01 = pd.DataFrame([i**2 for i in dataframe_01.values])


dataframe_01 = pd.DataFrame(np.arange(1,21).reshape((5,4)),index=range(0,5,1), 
                            columns=list('ABCD'))
dataframe_01


#################
# eval() method #
#################
# High performance memory-saving operation

# Create new column according to operations
dataframe_01.eval('E = A + B+C', inplace = False)
dataframe_01.eval('F = A**2', inplace = False)


#==========================#
# multiply df times vector #
#==========================#

import pandas as pd

import numpy as np

dataframe_01 = pd.DataFrame(np.arange(1,21).reshape((5,4)),index=range(0,5,1), 
                            columns=list('ABCD'))
dataframe_01

# multiply each row value times one number
y=[10,10,10,10, 100]
dataframe_01.multiply(y, axis=0)

# multiply each column value times one number
y=[10,10,10,100]
dataframe_01.multiply(y, axis=1)


#####################
# Create crosstable #
#####################

import pandas as pd
import numpy as np

# Create 2 variables
city_1 = (['sunny','cloudy','rainy','rainy','sunny','sunny','cloudy','rainy','rainy','sunny'])
city_2 = (['rainy','cloudy','rainy','cloudy','sunny', 'cloudy','rainy','cloudy', 'cloudy','sunny' ])
# Merge data into ndarray
array_01 = np.array([city_1,city_2]).T
array_01
df_01 = pd.DataFrame(array_01)

# Create joint probabilities and margin probabilities 
df_02 = pd.crosstab(df_01.iloc[:,0], df_01.iloc[:,1], margins=True, margins_name="Total")
df_02


# or

a = list('abbaabbabababbabbbabbbabbbabbab')
b = list('abbabbbabbbabbbabbbbbbabbbaaabb')
a2 = np.array([a,b]).T
a2
df = pd.DataFrame(a2,columns=['a','b'])
df2 = pd.crosstab(df.a,df.b, margins = True, normalize = True)
df2



###############################################################
# Sorting #
#!#########

dataframe_07 = pd.DataFrame(np.random.randint(0,10,100).reshape(25,4))

# Sort dataframe by choosen column
dataframe_07.sort_values(by=2, # sort by third column with name '2'
                         axis = 0, # sort rows, ('1' - sort columns) 
                         ascending=True,
                         inplace=False, # not overwrite current df with results
                         kind='quicksort', # quicksort method
                         na_position='last') # situate n/a as the last

# Sort df by choosen list of columns
dataframe_07.sort_values(by=[2,3], 
                         axis = 0, 
                         ascending=True)
# or
dataframe_07.sort_values(by='column_name', axis = 0, ascending=True)

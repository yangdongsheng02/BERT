# list = [i**2 for i in range(1,11)]
# print(list)

# list = [i for i in range(1,21) if i%2==0]
# print(list)

# dict = {i: i**3 for i in range(1,6)}
# print(dict)

# words = ['apple', 'banana', 'cherry', 'date']
# dict = {i: len(i) for i in words  if len(i)>4}
# print(dict)

# def greet():
#     print("hello,world!")
#     return
# greet_test= greet()
# print(greet_test)

# def greet_name(name):
#     return "Hellow," + name +'!'
# greet_name_test=greet_name('Alice')
# print(greet_name_test)

# def add(a,b):
#     return a + b
# result = add(3,5)
# print(result)

# def power(base,exponent=2):
#     return base**exponent
# print(power(3),power(2,3))

# def min_max(numbers):
#     return min(numbers), max(numbers)
# print(min_max([3, 1, 4, 1, 5, 9, 2]))

# def sum_all(*args):
#     return sum(args)
# print(sum_all(1,2,3,4,5))

# def create_person(**person):
#     return person
# print(create_person(name="Bob", age=25, city="New York"))

# def square(x):
#     return x ** 2
# def double(x):
#     return x * 2
# def square_then_double(x):
#     return double(square(x))
# print(square_then_double(3))

# def remove_duplicates(lst):
#     new_lst = []
#     for item in lst:
#         if item not in new_lst:
#             new_lst.append(item)
#     return new_lst
# print(remove_duplicates([3, 1, 2, 3, 4, 2, 1]))

# def count_chars(s):
#     dict = {}
#     for char in s:
#         if char in dict:
#             dict[char] += 1
#         else:
#             dict[char] = 1
#     return dict
# print(count_chars("hello world"))

def group_by_length(words):
    dict = {}
    for w in words:
        length = len(w)
        if length not in dict:
            dict[length] = [w]   #当第一次遇到某个长度时，必须初始化一个列表，并将当前单词作为第一个元素加入。
        else:
            dict[length].append(w)
    return dict

print(group_by_length(['apple', 'bat', 'cat', 'elephant', 'dog']))

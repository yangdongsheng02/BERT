# fruits = ['apple', 'banana', '火龙果','葡萄','芒果']
# fruits.append('西瓜')
# fruits.insert(1,'香蕉')
# print(fruits)
# del fruits[2]
# print(fruits)

# list_new = []
# sum = 0
# matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# for element in matrix:
#     for i in element:
#         if i % 2 == 0:
#             list_new.append(i)
# print(list_new)
# for element in matrix:
#     for i in element:
#        sum += i
# print(sum)

# numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
# list1 =[]
# numbers.sort()
# print(numbers)
# for number in numbers:
#     if number not in list1:
#         list1.append(number)
# print(list1)
# print(max(numbers))
# print(min(numbers))

# scores = [85, 92, 78, 90, 88, 76, 95, 89, 84, 91]
# total = 0
# lag_avg = []
# i = 0
# for score in scores:
#     total += score
#     avg = total / len(scores)
# print(avg)
# for score in scores:
#     if score > avg:
#         lag_avg.append(score)
# print(lag_avg)
# scores.sort()
# scores.reverse()
# print(scores)
# for score in scores:
#     if score > 90:
#         i += 1
# print(i)

# tuple = ('刘备','关羽','张飞','赵云','黄忠')
# print(len(tuple))
# print(tuple[2])
# print(tuple[-1])
# print(tuple.count('诸葛亮'))

# student = ("张三", 18, "男", 90.5)
# name,age,sex,score = student
# print(name,age,sex,score)

# numbers = (1, 2, 3, 4, 5)
# #numbers[0] = 10
# numbers.append(6)
# print(numbers)
# numbers[0] = 10
# 错误信息：TypeError: 'tuple' object does not support item assignment
# 原因：元组是不可变（immutable）的数据类型。Python中的元组一旦创建，其元素的值和数量就不能被修改。
# numbers.append(6)
# 错误信息：AttributeError: 'tuple' object has no attribute 'append'
# 原因：元组没有append()方法。append()是列表（list）的专属方法，用于动态修改列表内容。元组作为不可变类型，不支持任何会改变其元素数量或值的方法（如append、extend、pop等）。

# fruits = ("苹果", "香蕉", "橙子", "苹果", "葡萄", "香蕉", "苹果")
# print(fruits.count('苹果'))
# print(fruits.index('香蕉'))
# print(len(fruits))

# colors = ["红色", "绿色", "蓝色", "黄色", "紫色"]
# colors = tuple(colors)
# print(colors)
# print(colors[0:3:1])
# colors = list(colors)
# colors.append('橙色')
# print(colors)
# colors = tuple(colors)
# print(colors)

# student = {"name": "张三", "age": 18, "gender": "男", "score": 90}
# student['class'] = '一班'
# student['age'] = 19
# print(student)
# del student['gender']
# print(student)

# fruit_prices = {"苹果": 5.5, "香蕉": 3.0, "橙子": 4.2, "葡萄": 6.8}
# sum_prices = 0
# for key in fruit_prices.keys():
#     print(key,end='\t')
# for price in fruit_prices.values():
#     sum_prices += price
#     avg_price = sum_prices/len(fruit_prices)
# print('\n',avg_price)

# students = {
#     "001": {"name": "李华", "scores": {"math": 85, "english": 92, "science": 78}},
#     "002": {"name": "王明", "scores": {"math": 90, "english": 88, "science": 95}}
# }
# print(students["001"]['scores']['english'])
# students["002"]['scores']['history'] = 97
# print(students)

# employees = [
#     {"id": "E1001", "name": "张三", "department": "技术部", "salary": 8000},
#     {"id": "E1002", "name": "李四", "department": "市场部", "salary": 7500},
#     {"id": "E1003", "name": "王五", "department": "技术部", "salary": 8500}
# ]
# total= 0
# employees.append({"id": "E1004", "name": "赵六", "department": "财务部", "salary": 7000})
# print(employees)
# for employee in employees:
#     total += employee["salary"]
#     avg = total / len(employees)
# print(avg)

# numbers = [1, 2, 2, 3, 4, 4, 4, 5, 5, 6, 7, 7, 8, 9, 9, 9]
# count = -1
# number1 = []
# set1 = set(numbers)
# set2 = set()
# print(set1)
# print(f'原列表中有{len(set1)}种不同的数字')
# for number in numbers:
#     if number in set1:
#         count += 1
#         if count > 2:
#             set2.add(number)
# print(f"列表中所有出现次数超过2次的数字为{set2}")

list1 = [i**2 for i in range(1,11)]
print(list1)
list2 = [i**2 for i in range(1,21) if i%3 == 0]
print(list2)
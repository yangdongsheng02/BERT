# str = 'python is easy~'
# for i in str:
#     print(i)

# str = 'ok, i say ok! ok?'
# print(str.count('k'))

# count = 0
# for i in range(31,55+1,1):
#     count += i
# print(count)

# count = 0
# for i in range(100,201,1):
#     if i % 7 == 0:
#         count += i
# print(count)

# sum = 0
# count = 0
# for i in range(100,201,1):
#     if i % 7 == 0:
#         count += 1
#         if count  == 3:
#             continue
#         sum += i            #先判断是否是第三次再累加
# print(sum)

# str = "http://www.jd.com"
# print(str[7::])

# str = "music.avi"
# print(str[-3::])

# email = "zhangsan123@gmail.com"
# i = email.find('@')
# print(email[:i])
# print(email[i+1:])

# article = "人工智能正在改变世界。人工智能技术发展迅速，人工智能将带来新的革命。"
# print(article.count('人工智能'))
# print(article.replace('人工智能', 'AI'))

# contacts = "张三-13800138000;李四-13900139000;王五-13700137000"
# contacts=contacts.replace(';', ',')
# contacts=contacts.replace('-', ':')
# print(contacts)

# fruits = ['apple', 'banana', 'orange', 'grape', 'banana']
# print(fruits.index('banana'))
# print(fruits.count('banana'))
# if 'mango' in fruits:
#     print('在')
# else:
#     print('不在')

# students = ['张三', '李四', '王五']
# students.append('赵六')
# students.insert(2,'钱七')
# students.extend(['孙八', '周九'])
# print(students)

# scores = [85, 92, 78, 95, 88]
# scores.sort()
# print(scores)
# # scores.sort(reverse=True)
# scores.reverse()
# print(scores)

# cart = ['牛奶', '面包', '鸡蛋', '水果', '饼干']
# del cart[2]
# print(cart)
# cart1=cart.pop(1)
# print(cart1)
# cart.remove('水果')
# print(cart)

# characters = ['战士', '法师', '射手']
# characters.insert(3,'刺客')
# print(characters)
# characters.extend(['辅助', '坦克'])
# print(characters)
# print(characters.pop(-1))
# print(characters)

# numbers = [1, 3, 5, 7, 9, 2, 4, 6, 8]
# print(numbers.index(5))     #如果已知索引（下标）想获取值 → 直接用列表[索引]
# if 10 in numbers:           #如果已知值想获取索引（下标） → 用列表.index(值)
#     print('Ture')
# else:
#     print('False')
# print(numbers.count(3))
# numbers.count(3)

# todo_list = ['学习Python', '写作业', '锻炼身体']
# todo_list.append('阅读书籍')
# print(todo_list)
# todo_list.insert(1,'吃午饭')
# print(todo_list)
# todo_list.remove('锻炼身体')
# print(todo_list)

# prices = [100, 200, 300, 400, 500]
# for i in range(len(prices)):      #range(len(list)),按照长度来遍历 i =下标数字从0开始,直接遍历list是便利内容
#     prices[i] = prices[i] * 0.8
#     print(prices[i])
# prices.sort()
# print(prices[-1])

students = ['张三', '李四', '王五', '赵六']
scores = [85, 92, 78, 88]
# 1. 使用for循环计算平均分
avg_score = 0
for score in scores:
    avg_score = sum(scores) / len(scores)
print(avg_score)
# 2. 找出高于平均分的学生名单
list1 = []
for score in scores:
    if score > avg_score:
        i = scores.index(score)
        list1.append(students[i])
print(f'高于平均分的学生为{list1}')
# 3. 添加一个新学生'钱七'，成绩95
students.append('钱七')
scores.append(95)
# 4. 删除成绩最低的学生
min_score = min(scores)
print(min_score)
i = scores.index(min_score)
print(i)
del scores[i]
del students[i]
print(students)
print(scores)

# 5. 将所有成绩提高5分（最高不超过100）
for i in range(len(scores)):
    scores[i] = scores[i] +5
    print(scores)


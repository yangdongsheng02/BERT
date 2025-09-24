# 1. 利用字典, 统计字符串中每个字符的次数
# 	需求: 用户输入一个字符串my_string
# 	用户字典统计每个字符出现的次数
my_string = 'aabbbCCCC111122222'
my_dict = {}
for char in my_string:
    if char in my_dict:
        my_dict[char] += 1
    else:
        my_dict[char] = 1
print(my_dict)


# 2. 字符串转字典
# 	编写一个程序将字符串转换为字典
# 	'name=坤坤&age=18&desc=唱跳rap篮球'
# 	{ 'name': '坤坤',   'age':  '18',   'hobby':  '唱跳rap篮球' }
name = 'name=坤坤&age=18&desc=唱跳rap篮球'
my_list = []
my_dict1 = {}
my_list=name.split('&')
print(my_list)
for item in my_list:
    k,v = item.split('=')
    my_dict1[k] = v
print(my_dict1)


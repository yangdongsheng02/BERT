# original_list = [1,2,2,2,4,4,5,6,6,7]
# unique_list = []
# for i in original_list:
#     if i not in unique_list:
#         unique_list.append(i)
# print(unique_list)

# nested_tuple = ((1,2,3,4),(2,4,6),(2,3,5))
# unique_elements = set()#空集合
# for i in nested_tuple:#这里的i是元组(大元组里的小元组)
#     for j in i:
#         unique_elements.add(j)
# unique_list = list(unique_elements)#集合转换为列表
# print(unique_list)
#
# my_tuple = (1,2,3,4,5,6,7,8,9)
# count = 0
# for i in my_tuple:
#     if (i % 2) != 0 :
#         count = count + 1
# print(count)
#
#
my_string = "hello world"
person = {}
for char in my_string:
    if char in person:
        person[char] += 1 #第一次遇到字符 'h',字典是空的，没有'h'这个键跳到下面创建键'h'，并设置值为1
    else:
        person[char] = 1    #设置值为1
for char, count in person.items():
    print(f"字符 '{char}' 出现了 {count} 次")


# input_string = input("请输入一个字符串: ")
# result_dict = {}
# key_value_pairs = input_string.split()
# for pair in key_value_pairs:
#     key, value = pair.split('=')
#     result_dict[key] = value
# print(result_dict)

from numpy import number

# for i in range(2023,2026):
#     for j in range(1,12):
#         print(f'{i}年{j}月')

num1 = {}
num1 =  {i: i**2 for i in range(1,6)}
print(num1)


#姓名和年龄列表,组成一个字典
name_list = ['乔峰','虚竹','段誉']
age_list = [38,28,18]
users = {name: age for name,age in zip(name_list,age_list)}
users1 = {name_list[i]: age_list[i] for i in range(len(name_list))}
print(users)
print(users1)
# numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# count = 0
# for i in numbers:
#     if i % 2 != 0:
#         count += i
# print(count)


n = int(input("输一个数"))
num_list = [i for i in range(1,n+1)]
current_num = 0 #当前
i = 0  #下标

while len(num_list) !=1:   #多于一个人就继续
    current_num += 1
    if current_num % 3 == 0:
        num_list.pop(i)   #删除下标为i的元素
        i -= 1
    i += 1
    if i >= len(num_list):
        i = 0
print(f'{n}人参与游戏, 幸运数字是: {num_list[0]}')


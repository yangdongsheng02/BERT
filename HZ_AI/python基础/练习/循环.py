# #while循环
# #while循环求1到100的和
# i = 1
# result = 0
# while i <= 100:
#     result = result + i
#     i = i + 1
# print(result)


# #我自己的代码
# i = 0
# result = 0
# while i<=100:
#     i=i+1
#     if i % 2 == 0:
#         result += i
# print(result)
#
#
# # 初始化计数器（老师代码）
# i = 1
# # 定义result，用于接收所有偶数的和
# result = 0
# # 编写循环条件
# while i <= 100:
#     # 将来写代码的位置
#     if i % 2 == 0:
#         # 代表变量i是一个偶数
#         result += i
#     # 更新计数器
#     i += 1
# print(f'1~100之间所有偶数的和：{result}')

# # 初始化计数器
# i = 1
# # 编写循环条件
# while i <= 5:
#     # 当变量i == 3的时候，中止当前循环，继续下一次循环
#     if i == 3:
#         # 手工更新计数器(非常重要)
#         i += 1
#         print('吃到了一只大虫子，这个苹果不吃了...')
#         continue
#
#     print(f'正在吃第{i}个苹果')
#     # 更新计数器
#     i += 1

# sum = 0
# for i in range(1,101,1):
#     sum += i
# print(sum)

sum = 0
for i in range(1,101,1):
    if i % 2 == 0:
        sum += i
print(sum)

count = 1
for i in range(1,11,1):
    count *= i
print(count)

count1 = 1
for i in range(10,31,1):
    if i % 7 != 0:
        count1 += i
print(count1)


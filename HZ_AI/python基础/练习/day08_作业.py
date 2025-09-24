# age = int(input('输入年龄'))
# if age >= 18:
#     print('已成年')
# else:
#     print('未成年')
from sympy.physics.units import amount

# number = int(input("输入一个数字"))
# if number == 0:
#     print('零')
# elif number >0:
#     print('正数')
# else:
#     print('负数')

# score = int(input('输入分数'))
# if score >= 90:
#     print('优秀')
# elif score >= 80:
#     print('良好')
# elif score >= 70:
#     print('中等')
# elif score >= 60:
#     print('及格')
# else:
#     print('不及格')


count = 3
for i in range(3):
    username = input("输入用户名")
    password = int(input('密码'))
    if username == 'admin':
        if password == 123456:
            print('登陆成功')
            break
        else:
            print('密码错误')
            count = count - 1
            print(f'还剩{count}次机会')
    else:
        print('账号错误')
        count = count - 1
        print(f'还剩{count}次机会')


# amount = int(input('输入购物金额'))
# if amount >= 1000:
#     amount = amount*0.8
#     print(f'您好,你一共{amount}元')
# elif amount >=500:
#     amount = amount*0.9
#     print(f'您好,你一共{amount}元')
# elif amount >=200:
#     amount = amount*0.95
#     print(f'您好,您一共{amount}元')
# else:
#     print(f'您好,你一共{amount}元')


# i = 50
# count = 0
# while i <= 100:
#     count += i
#     i += 1
# print(count)


# i = 1
# count = 0
# while i <=100:
#     if i%5 ==0:
#         i +=1
#         continue
#     count += i
#     i += 1
# print(count)


# i = 1
# count = 0
# j = 0
# while i <=100:
#     if i%7 == 0:
#         j += 1
#         if j == 3:
#             break
#         else:
#             count += i
#     count += i
#     i += 1
# print(count)

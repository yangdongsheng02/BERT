# str1 = 'ithema'
# for i in str1:
#     print(i)

# #案例: 使用for循环，求1 ~ 100之间所有偶数的和
# result = 0
# for i in range(101): #rabge()包左不包右
#     if i % 2 == 0:
#         result += i
# print(f'1到100偶数和为{result}')


# 案例：用for循环实现用户登录
# ① 输入用户名和密码
# ② 判断用户名和密码是否正确（username='admin'，password='admin888'）
# ③ 登录仅有三次机会，超过3次会报错
# 分析：用户登陆情况有3种:
# ① 用户名错误(此时便无需判断密码是否正确)  -- 登陆失败
# ② 用户名正确 密码错误 --登陆失败
# ③ 用户名正确 密码正确 --登陆成功

# trycount = 0
# for i in range(3):
#     username = input("输入登录账号")
#     password = input("输入密码")
#     trycount = trycount + 1
#     if username == 'admin' and password == 'admin888':
#         print("登录成功")
#     else:
#         print("登录失败")
#         print(f'剩余{3-trycount}登录次数')
#我自己的不完整

#老师代码
trycount = 0
for i in range(3):
    username = input("输入登录账号")
    password = input("输入密码")
    trycount = trycount + 1
    if username == 'admin':
        if password == 'admin888':
            print("登录成功")
            break #没有 break 时，无论登录是否成功，循环都会执行满 3 次（因为 range(3) 控制了循环次数为 3 次）。
        else:
            print("密码错误")
            print(f'剩余{3 - trycount}登录次数')
    else:
        print("用户名错误")
        print(f'剩余{3-trycount}登录次数')
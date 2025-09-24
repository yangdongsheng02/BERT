# #我的第一版思路,遍历100-999符合水仙花数则cout+1
# import random
# count=0
# for i in range(100,1000):
#      if i == (i // 100)**3 + (i // 10 )**3 + (i % 10)**3:
#         count+=1
# print(count)
#全错QAQ

count = 0
for i in range(100,1000):
    a = i // 100
    b = (i // 10)%10
    c = i % 10
    if i == a**3 + b**3 + c**3:
        count += 1
print(count)


#思路for循环设置三次机会,if判断账号密码输入对不对
ci = 0
for i in range(3):
    username = input("请输入账号")
    password = input("输入密码")
    ci += 1
    if username == 'itcast':
        if password == 'hmai':
            print("登陆成功")
            break #如果不用break登录成功也会三次循环完
        else:
            print('密码错误')
            print(f'剩余{3-ci}次登陆机会')
    else:
        print("账号错误")
        print(f'剩余{3-ci}次登陆机会')





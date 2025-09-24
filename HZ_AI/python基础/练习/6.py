# 编写一个Python程序，随机生成一个1到100之间的整数，用户通过输入猜测的数字，
# 程序会根据用户的输入提示“猜大了”、“猜小了”或“猜对了”。用户可以无限次猜测，
# 直到猜对为止。猜对后，程序会提示用户并结束游戏。

import random
guess_num = random.randint(1,100)
print(guess_num)
while True:   #因为不知道用户多少次能猜对, 用 while循环.
    input_num = int(input('输个数'))
    if guess_num == input_num:
        print('猜对了')
        break
    elif guess_num > input_num:
        print('猜小了')
    else:
        print('猜大了')

#两次
# i = 1
# while i <= 10:
#     if i % 3 == 0:
#         break
#         print('hello world')
#     print(f'hello world {i}')
#     i += 1


# #7次
# i = 1
# while i <= 10:      # i的值: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#     if i % 3 == 0:
#         i = i + 1
#         continue   # 7次 相当于少了三次，当i能被3整除时不打印
#     print(f'hello world {i}')
#     i += 1


# i = 1
# while i <= 10:      # i的值: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#     if i % 3 == 0:
#         print(f'hello world {i}')   # 13次，多三次，i被三整除时多打印一次
#     print(f'hello world {i}')
#     i += 1
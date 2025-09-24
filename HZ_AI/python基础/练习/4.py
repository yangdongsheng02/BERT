# num1 = 10
# num2 = 2
#
# # 四则运算 + - * /
# print(f'加：{num1 + num2}')
# print(f'减：{num1 - num2}')
# print(f'乘：{num1 * num2}')
# print(f'除：{num1 / num2}')
#
# total_friends = int(input())
# total_bill = float(input())
# num = total_bill*1.2
# amount_per_person = num / total_friends
# print(f"每位顾客应付：{amount_per_person:.2f}元")
#
# age = int(input('age'))
# if age > 18:
#     print("True")
# else:
#     print("False")
from pyarrow import nulls
#
# limit = int(input('高速速度'))
# print(f'速度必须在60~120{limit>=60 and limit<=120}')

#判断用户是否为空
username = input('输入用户名')
print(f'非空:{username != ''}')
if not username:
    print('用户名不能为空')
else:
    print('输入正确')
    

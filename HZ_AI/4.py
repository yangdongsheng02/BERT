num1 = 10
num2 = 2

# 四则运算 + - * /
print(f'加：{num1 + num2}')
print(f'减：{num1 - num2}')
print(f'乘：{num1 * num2}')
print(f'除：{num1 / num2}')

total_friends = int(input())
total_bill = float(input())
num = total_bill*1.2
amount_per_person = num / total_friends
print(f"每位顾客应付：{amount_per_person:.2f}元")
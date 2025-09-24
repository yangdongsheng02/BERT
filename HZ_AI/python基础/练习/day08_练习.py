# score = int(input('输入成绩'))
# if score >= 60:
#     print('成绩合格')
# else:
#     print('成绩不合格')
#
# price = int(input('兜里零花钱'))
# if price > 100:
#     print("吃大餐")
# else:
#     print('泡面')
#
# price1 = int(input("price1"))
# if price1> 10000 :
#     print('海南三天')
# elif price1> 1000 :
#     print('迪士尼一天')

#
# for i in range(2):
#     job = input('选择你的职业')
#     mana = int(input('你的法力值'))
#     if job == '法师':
#         if mana >= 100:
#             print('大火球术')
#         elif mana >= 50:
#             print('冰冻雨')
#         else:
#             print('闪电箭')
#     elif job == '法师':
#         if mana >= 100:
#             print('旋风斩')
#         elif mana >= 50:
#             print('战斧投掷')
#         else:
#             print('怒砍一刀')
#     else:
#         print("请选择'战士'或'法师'")
# i = 0
# while i < 5:
#     print('hello python')
#     i += 1
# i=1
# count = 0
# while i<101:
#     count += i
#     i += 1
# print(count)

# i=1
# count = 0
# while i<101:
#     if i%2 == 0:
#         count = count + i
#         i = i+1
#     else:
#         i = i+1
# print(count)

# i = 1
# count = 1
# while i < 11:
#     count = count * i
#     i += 1
# print(count)


# i = 1
# count = 0
# while i<=20:
#     if i%3 == 0:
#         i += 1
#     else:
#         count += i
#         i += 1
# print(count)

i = 1
count = 0
while i<=20:
    if i%7 == 0:
        break
    else:
        count += i
        i += 1
print(count)

i = 1
count = 0
while i<=20:
    if i%3 == 0:
        i = i+1
        continue
    count += i
    i += 1
print(count)
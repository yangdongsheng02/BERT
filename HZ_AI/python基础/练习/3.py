num= input('输个数:')
print(type(num))

print('-'*20)  #输出二十遍-

num = int(num)
print(type(num))

num = int(input('输入一个数：'))
print(type(num))

AB = int(input('上底'))
DC = int(input('下底'))
h = int(input('高'))
S = ((AB+DC)/2)*h
print('梯形的面积为:%.2f'%(S))
print(f'梯形的面积为:{S}')


#测试
str1 = 'apple-banana-orange'
print(str1.split('-'))

list1 = ['apple', 'banana', 'orange']
print('-'.join(list1))

s = input('输入一个字符串')
n = int(input('要旋转几位'))
n = n % len(s)
rotated_string = s[-n:] + s[:-n]
print(f'旋转后为"{rotated_string}"')
numbers = [2,5,3,7,5,7]
count = 0
min = min(numbers)
max = max(numbers)
number_set = set()
for i in range(min,max):
    if i not in numbers:
        number_set.add(i)
        count += i
print(f'缺失数字为{number_set},总和为{count}')

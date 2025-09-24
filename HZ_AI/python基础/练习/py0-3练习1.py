numbers = [7,8,9,5,6,7,2,3]
max_values =  []
for i in range(len(numbers)-1):
    if numbers[i] > numbers[i+1]:
        a = numbers[i]
        max_values.append(a)
    else:
        b = numbers[i+1]
        max_values.append(b)
print(max_values)

#简化
numbers1 = [7,8,9,5,6,7,2,3]
max_values =  []
for i in range(len(numbers1)-1):
        a = numbers1[i]
        b = numbers1[i+1]
        c = max(a,b)
        max_values.append(c)
print(max_values)

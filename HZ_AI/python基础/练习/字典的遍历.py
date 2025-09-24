sorce = {'数学':85,
            '语文':92,
            '英语':78,
            '物理':88,
            '化学':95}
for key in sorce.keys():
    print(key)
total = 0
for value in sorce.values():
    total += value
    avg = total / len(sorce)
print(avg)

for k,v in sorce.items():
    print(k,v)
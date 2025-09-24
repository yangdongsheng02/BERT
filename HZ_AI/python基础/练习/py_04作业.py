def calculator(a,b):
    jia = a+b
    jian = a-b
    cheng = a*b
    chu = a/b
    return jia, jian, cheng, chu
a = int(input("输入第一个数"))
b = int(input("输入第二个数"))
print(calculator(a,b))


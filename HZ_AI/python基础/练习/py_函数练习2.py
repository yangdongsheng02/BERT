a = 10
print(id(a))

def fn1():
    return 100

# 调用fn1函数
print(fn1)  # 返回fn1函数在内存中的地址#直接打印fn1,而fn1中存的是fn1函数在内存中的地址
print(fn1())  # 代表找到fn1函数的地址并立即执行

fn2 = lambda num1, num2: num1 + num2
print(fn2(10, 20))

fn = lambda a, b : a if a > b else b

print(fn(10, 20))
# get_cheng = lambda a,b: a+b
# print(get_cheng(10,20))

def my_calculate(my_func,a,b):
    return my_func(a,b)
get_sum = lambda a,b : a+b
print(my_calculate(get_sum,10,20))

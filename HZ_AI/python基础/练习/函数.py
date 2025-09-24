# def ger_num(n1,n2,n3):
#     result = n1*n2*n3
#     return result
# test = ger_num(1,2,3)
# print(test)

# def look(n):
#     for i in range(n):
#         print("注意看,这个男人叫小帅~")
#     return
# look(2)
# print('-----')
# result = look(5)
# print(result)



my_list = [i for i in range(11)]
def get_num(list):
    '''
    求列表中奇数,偶数的和
    :param list:
    :return:
    '''
    sum1 = 0
    sum2 = 0
    for i in my_list:
        if i%2 == 1:
            sum1 += i
        else:
            sum2 += i
    return [sum1, sum2]
    print(f'奇数和为{sum1},偶数和为{sum2}')

test = get_num(my_list)
print(test)
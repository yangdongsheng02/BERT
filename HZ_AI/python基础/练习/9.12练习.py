# scores = [81,82,83,84]
# for score in scores:
#     total = sum(scores)
# print(total)
#
# list1 = []
# prices = [1000,2000,3000,4000,5000]
# for price in prices:
#     if price > 3000:
#         list1.append(price)
# print(list1)
#
# students = ['刘备','张飞','韩信']
# for i in range (len(students)):
#     print(students[i],i)



# heros = [
#     ['关羽','亚瑟','吕布'],
#     ['西施', '小乔', '安琪'],
#     ['伽罗', '黄忠', '后裔'],
#     ['明世隐', '大乔', '盾山'],
#     ['澜', '镜', '霏']
# ]
# for hero in heros:
#     for item in hero:
#         print(item,end='\t')
#     print('',end='\n')


# for hero in heros:
#     for item in hero:
#         print(item,'\t')  #而第二个循环，由于内层循环的每个print都会换行，所以每个item都会单独占一行，导致输出变成垂直排列。
#     print('','\n')        #循环中的print(item, '\t')实际上等同于print(item, end='\n')，


list1 = [7,8,9,5,6,7,2,3]
list2 = []
for i in range (len(list1)-1):
    value = max(list1[i],list1[i+1])
    list2.append(value)
print(list2)

original_list = [1,2,2,3,4,4,5,6,6,7]
list3 = []
for item in original_list:
    if item not in list3:
        list3.append(item)
print(list3)


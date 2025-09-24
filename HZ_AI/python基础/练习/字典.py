# user = {"姓名":'刘总',
#         '资产':'2000万',
#         '性别':'男',
#         '年纪':'30'}
# print(user)
# telepone = {
#     '刘总':'13877776666',
#     '肖总':'13911118888',
#     '李总':'13533334444'
# }
# print(telepone)

students = {'s001':'张三',
            's002':'李四',
            's003':'王五',
            's004':'赵六'}
print(students)
print(students['s002'])
print(students.keys())
print(students.values())
print(students.items())
students['s005'] = '田七'
print(students)
students['s002'] = '李小龙'
print(students)
del students['s004']
print(students)
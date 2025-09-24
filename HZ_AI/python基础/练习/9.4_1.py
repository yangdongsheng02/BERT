numstr = '0123456789'
print(numstr[2:5:1])
print(numstr[2:5])
print(numstr[:-1])
# 6、把步阶设置为负整数：类似字符串翻转
print(numstr[::-1])

filename = 'avatar.png'
# 获取点号的索引下标
index = filename.find('.')
print(index)
# 使用切片截取文件的文件
name = filename[:index]
print(f'上传文件的名称：{name}')

# 使用切片截取文件的后缀
postfix = filename[index:]
print(f'上传文件的后缀：{postfix}')


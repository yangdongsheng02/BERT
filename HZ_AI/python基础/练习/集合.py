comment_tags = ['python','编程','学习','python','代码','学习','算法']
print(set(comment_tags))

permissions = {'read','write'}
permissions = set(permissions)
print(permissions)
permissions.add('delete')
permissions.add('add')
permissions.add('read')
permissions.add('write')
permissions.remove('write')
print(permissions)
print('add' in permissions)


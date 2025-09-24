def print_info():
    print('1. 添加学生信息')
    print('2. 删除学生信息')
    print('3. 修改学生信息')
    print('4. 查询单个学生信息')
    print('5. 查询所有的学生信息')
    print('6. 退出系统')

students = [] #学生列表

#添加学生
def add_student():
    id = int(input('输入学号'))
    if id in students:
        print('学号已存在')
    name = input('输入学生名字')
    age = int(input('输入年龄'))
    stu_dict = {'id': id, 'name': name, 'age': age}
    students.append(stu_dict)

#删除学生
def del_student():
    i = int(input("请输入要删除学生的学号"))
    for stu in students:
        if stu['id'] == i:
            students.remove(stu)
            print('已删除')
            break
    else: #else缩进与for循环对齐，属于for循环的一部分。当循环执行完毕（即遍历完所有学生后仍未找到匹配的学号），else块中的代码会执行，打印“学号不存在”。而如果在循环中找到匹配项并执行break，则else块不会执行。
        print('学号不存在')

#修改学生信息
def update_student():
    i = int(input("输入要修改的学生学号"))
    for stu in students:
        if stu['id'] == i:
            new_id = int('输入学号')
            new_name=int(input("名字"))
            new_age = int(input("输入年龄"))
            stu['age'] = new_age
            stu['id'] = new_id
            stu['name'] = new_name
            print('修改完成')
    else:
        print('学号不存在')

#查询单个学生信息
def select_student():
    t = int(input('输入要查询学生学号'))
    for stu in students:
        if stu['id'] == t:
            print(stu)
            break
    else:
        print('学号不存在')

 #展示所以学生
def show_student():
    for stu in students:
        print(stu)

while True:
    print_info()
    num = int(input('输入操作数字'))
    if num == 1:
        add_student()
    elif num == 2:
        del_student()
    elif num == 3:
        update_student()
    elif num == 4:
        select_student()
    elif num == 5:
        show_student()
    elif num == 6:
        print("感谢使用,下次见")
        break
    else:
        print("请输入对应数字")
    print()
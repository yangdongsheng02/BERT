class Car:
    def __init__(self,color):
        #设置属性 __int__
        self.color = color


    #     #获取属性:self.属性名
    # def show(self):
    #     print(self.color)
    #     print(self.number)

    #del魔法
    def __del__(self):
        print('自动调用了del魔法方法')

#创建对象:对象名 = 类名()
car = Car('红色')

del car
print(car.color)
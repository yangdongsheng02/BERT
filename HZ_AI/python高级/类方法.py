# 1.定义类
class Dog(object):
    __num = 6
    # 类方法
    @classmethod
    def eat(cls):
        print(cls.__num)
        print('小狗喜欢吃骨头')

# 2.调用类方法
Dog.eat()
# dog = Dog()
# dog.eat()
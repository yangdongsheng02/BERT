class Student:
    def __init__(self):
        self.weight = 100

    def run(self):
        self.weight -= 0.5
        print(f'跑步一次之后体重为{self.weight}')

    def eat(self):
        self.weight += 0.5
        print(f'大吃大喝后体重为{self.weight}')

xiaoming = Student()
xiaoming.run()
xiaoming.eat()

#烤地瓜



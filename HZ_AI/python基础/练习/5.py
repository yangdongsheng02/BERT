proof = int(input('请输入驾驶员100ml血液中的酒精含量：'))
if proof <= 20:
    print("没事")
else:
    if proof < 80:
        print("酒驾")
    else:
        print("醉驾")


import random
playre = int(input("请输入您的出拳0-代表石头，1代表剪刀，2代表布"))
computer = random.randint(0,2)
if (playre == 0 and computer == 1) or (playre == 1 and computer == 2) or (playre == 2 and computer == 0):
    print("player win")
elif playre == computer:
    print("平")
else:
    print("computer win")

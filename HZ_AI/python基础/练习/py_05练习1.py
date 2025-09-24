# 1、打开文件
f = open('python.txt', 'r+')

lines = f.readlines()
for line in lines:
    print(line, end='')
# 3、关闭文件
f.close()

import os
import time
os.rename('python.txt', 'linux.txt')
time.sleep(20)

os.remove('linux.txt')
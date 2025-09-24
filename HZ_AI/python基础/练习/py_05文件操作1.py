# import os
# if not os.path.exists('../../images'):
#     os.mkdir('../../images')
#
# if not os.path.exists('images/avatar'):
#     os.mkdir('images/avatar')
#
# print(os.getcwd())
#
# os.chdir('images/avatar')
# print(os.getcwd())
#
# os.chdir('../../../../')
# print(os.getcwd())
#
# print(os.listdir())
# os.rmdir('images/avatar')

# dis_f = open("data/1.txt",'w',encoding='utf-8')
# dis_f.write("锄禾日当午\n")
# dis_f.write('汗滴禾下土')
# src_f = open("data/1.txt",'r',encoding='utf-8')
# while True:
#     byts = src_f.read(3)
#     if len(byts) == 0:
#         break
#     print(byts)
# src_f1 = open("data/1.txt",'r',encoding='utf-8') #按字节读不用指定编码格式
# while True:
#     byts = src_f1.read(5)
#     if len(byts) == 0:
#         break
#     print(byts)

# result = src_f1.readline()
# # result1 = src_f1.readline()
# print(result.replace('\n',''))
# print(result1)

# result2 = src_f1.readlines()
# for line in result2:
#     line = line.replace('\n','')
#     print(line)

# cat_f = open("data/cat.jpg",'rb')
# cat2_f = open("data/cat2.png",'wb')
# while True:
#     data=cat_f.read(1024)
#     if len(data) == 0:
#         break
#     cat2_f.write(data)
# cat_f.close()
# cat2_f.close()

src_name = '../data/1.txt'
idx = src_name.rfind('.')
dit_name = src_name[:idx] + '备份' + src_name[idx:]
with open(src_name,'rb') as src_f,open(dit_name,'wb') as dit_f:
    while True:
        data = src_f.read(1024)
        if len(data) == 0:
            break
        dit_f.write(data)
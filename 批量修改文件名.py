'''修改文件名和图片大小'''
import os
import re
from PIL import Image

path = r'C:\Users\TITAN\Desktop\KNN\data\\'

# for folder in os.listdir(path):
#     new_path = path + '\\' + str(folder) +'\\'
#     for item in range(860):#(len(os.listdir(new_path)))
#         oldname = new_path + os.listdir(new_path)[item]
#         name = new_path + str(folder) + '_' + str(item) + '.png'
#         if (name):
#             print('run')
#             os.rename(oldname,name)

# for folder in os.listdir(path):
#     new_path = path + '\\' + str(folder) +'\\'
#     for item in os.listdir(new_path):
#         file = os.path.join(new_path,item)
#         filename = item.split('.')[0]
#         print(filename)
#         if(re.search(r'[a-z]',filename)):
#             print('1')
#             os.remove(file)
        
for folder in os.listdir(path):
    new_path = path + '\\' + str(folder) +'\\'

    f_path = 'C:\\Users\\TITAN\\Desktop\\KNN\\test2\\' + str(folder)
    # if not exit(f_path):
    #     print('1')
    os.mkdir(f_path)
    for item in os.listdir(new_path):
        file = os.path.join(new_path,item)
        pic = Image.open(file)
        pic = pic.resize((50, 50))
        name = os.path.join(f_path,item)
        print(name)
        pic.save(name)


print('Finished!')
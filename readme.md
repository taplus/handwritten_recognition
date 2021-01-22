# 基于matlab采用KNN算法手写体数字识别实现

##  一、前言

- KNN 全称是 K- Nearest Neighbors ，K-近邻。简单来说，K就是邻居个数，选出和测试样本最像的邻居（这里是欧式几何距离最短的K个邻居），那么样本的邻居是什么，样本就是什么（在K个邻居里，假如邻居的标签最多的是数字1，我们就认为样本的标签就很可能是数字1）
- KNN 实现手写体识别的原理和代码都比较简单，但网上相关文章不多，本文只是把我自己的理解写下来作为学习matlab的实践，多有纰漏，请多指教

-------

##  二、实现过程

1. 处理 MNIST 数据集

    - 下载 [MNIST](http://yann.lecun.com/exdb/mnist/) 数据集，下载测试集、测试标签、训练样本、训练标签共四个文件
    - 下载下来的数据集是 IDX 文件格式的，因此用 Python 转为 50×50 的PNG图片，代码在后
    - 选取合适数量的测试集和训练集，训练集中每个数字的训练样本数要一致

2.  matlab 实现步骤（以图像分辨率为 50×50例）

   - 对所有图片做二值化处理：有值取1，无值取0

   - 将 0-9 数字的训练样本矩阵化，每一幅数字图像都是一维矩阵。以50×50分辨率图像为例，获得 1×2500 的一维矩阵；每个数字860张图片，我们就得到了 8600 × 2500 的矩阵，这作为训练矩阵
   - 在训练矩阵加入标签列，用来判断某一行指的数字是多少
   - 对每一幅待识别数字图像，同样将其转为 1 × 2500 的一维矩阵，称为测试矩阵
   - 计算测试矩阵与训练矩阵每一维的欧氏几何距离，同样按列向量加到训练矩阵，并按距离升序按行排列训练矩阵
   - 对前 K 个行向量求标签的众数，结果标签就是采用 KNN 算法得到的最有可能的识别结果

---


## 三、代码实现

1. **处理MINIST数据集的Python代码　感谢 [name_s_Jimmy](https://blog.csdn.net/qq_32166627) 的文章 [使用Python将MNIST数据集转化为图片](https://blog.csdn.net/qq_32166627/article/details/52640730)**

   ```python
   import numpy as np
   import struct
    
   from PIL import Image
   import os
    
   data_file =  #需要修改的路径，测试或训练样本图像，如t10k-images.idx3-ubyte或train-images.idx3-ubyte
   # It's 47040016B, but we should set to 47040000B
   data_file_size = 47040016
   data_file_size = str(data_file_size - 16) + 'B'
    
   data_buf = open(data_file, 'rb').read()
    
   magic, numImages, numRows, numColumns = struct.unpack_from(
       '>IIII', data_buf, 0)
   datas = struct.unpack_from(
       '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
   datas = np.array(datas).astype(np.uint8).reshape(
       numImages, 1, numRows, numColumns)
    
   label_file =  #需要修改的路径，测试或训练样本标签，如t10k-labels.idx1-ubyte或train-labels.idx1-ubyte
    
   # It's 60008B, but we should set to 60000B
   label_file_size = 60008
   label_file_size = str(label_file_size - 8) + 'B'
    
   label_buf = open(label_file, 'rb').read()
    
   magic, numLabels = struct.unpack_from('>II', label_buf, 0)
   labels = struct.unpack_from(
       '>' + label_file_size, label_buf, struct.calcsize('>II'))
   labels = np.array(labels).astype(np.int64)
    
   datas_root = r'C:\Users\TITAN\Desktop\KNN\test' #需要修改的路径
   if not os.path.exists(datas_root):
       os.mkdir(datas_root)
    
   for i in range(10):
       file_name = datas_root + os.sep + str(i)
       if not os.path.exists(file_name):
           os.mkdir(file_name)
    
   for ii in range(10000):# 生成10000张测试或训练样本
       img = Image.fromarray(datas[ii, 0, 0:50, 0:50])
       label = labels[ii]
       file_name = datas_root + os.sep + str(label) + os.sep + \
           'mnist_train_' + str(ii) + '.png'
       img.save(file_name)
   
   print('Finished!')
   ```

----

2. **Matlab 代码**

   ```matlab
   clc;
   clear;
   
   matrix = [];% 训练矩阵
   for delta = 0:9%构建训练区样本的矩阵
     label_path = strcat('C:\Users\ABC\Desktop\KNN\trian\',int2str(delta),'\');
     disp(length(dir([label_path '*.png'])));
     for i = 1:length(dir([label_path '*.png']))
           im = imread(strcat(label_path,'\',int2str(delta),'_',int2str(i-1),'.png'));
           %imshow(im);
           im = imbinarize(im);%图像二值化
           temp = [];
           for j = 1:size(im,1)% 训练图像行向量化
               temp = [temp,im(j,:)];
           end
           matrix = [matrix;temp];
     end
   end
   
   label = [];%在标签矩阵后添加标签列向量
    for i = 0:9
       tem = ones(length(dir([label_path '*.png'])),1) * i;
       label = [label;tem];
   end
   matrix = horzcat(matrix,label);%带标签列的训练矩阵
   
   %测试对象向量
   for delta = 0:9%构建测试图像的向量
       test_path = strcat('C:\Users\ABC\Desktop\KNN\test\',int2str(delta),'\');
       len = (length(dir([test_path '*.png'])));
       disp(len);
       p = 0;% 识别结果计数
       for i = 1:len
           vec = []; %　测试样本行向量化       
           test_im = imread(strcat('test2\',int2str(delta),'\',int2str(delta),'_',int2str(i-1),'.png'));
           imshow(test_im);
           test_im = imbinarize(test_im);
           for j = 1:size(test_im,1)
               vec = [vec,test_im(j,:)];
           end
   
           dis = [];
           for count = 1:length(dir([label_path '*.png'])) * 10
               row = matrix(count,1:end-1);% 不带标签的训练矩阵每一行向量
               distance = norm(row(1,:)-vec(1,:));% 求欧氏几何距离
               dis = [dis;distance(1,1)];% 距离列向量
           end
           test_matrix = horzcat(matrix,dis);% 加入表示距离的列向量
   
   
           %排序
           test_matrix = sortrows(test_matrix,size(test_matrix,2));
           %输入K值，前K个行向量标签的众数作为结果输出
           K = 5;
           result = mode(test_matrix(1:K,end-1));
           disp(strcat('图像',int2str(delta),'_',int2str(i),'.png','的识别结果是：',int2str(result)));
   
           if(delta == result)
               p = p + 1;
           end
           
           
       end
       pi = p/len;
       disp(strcat('识别精度为：',num2str(pi)));
       disp('Finished!'); 
   end
   ```

----


## 四、结果

- 采用 KNN (最近邻) 算法实现手写数字体的识别，经测试在 K = 5，训练样本 8600 的 条件下，总体精度在0.9以上，个别数字比如 8 识别就比较差只有 0.8 左右
- KNN 算法简单，但缺点也比较明显，运行时间长，容易收敛于局部值，精度不高。
- 提高训练样本数量，调整K值，在执行算法前对图像做初步处理可能会有更好的表现

----

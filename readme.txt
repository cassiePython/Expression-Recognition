第一步：
从http://mplab.ucsd.edu/wordpress/wp-content/uploads/genki4k.tar
下载数据集genki4k.tar；
解压到genki4k；
其中有一个files子目录，其中是样本图片；
另一个文件是labels.txt是样本的label

第二步：
使用extract_face.py进行面部的提取(包含转化为灰度图和resize为64*64)

第三步：
使用split_train_test.py进行训练集和测试集的划分

第四步：
构造训练集和测试集合的label：
data/test_labels.txt
data/train_labels.txt
其中格式为(样本名称+空格+标签(0或者1))：
        file0003.jpg 1
        file3948.jpg 0

第五步：
使用convert.py将训练集和测试集的图片转化为tfrecord的格式。

第六步：
使用train.py进行训练，模型会存储到path/to/model路径下

第七步：
使用evaluate.py输出测试结果

第八步：
check.py实现对单张图片的测试。
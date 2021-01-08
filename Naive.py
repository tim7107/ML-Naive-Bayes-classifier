#----------------------------------------------------------------------
#----------------------------Import_Library----------------------------
#----------------------------------------------------------------------
import numpy as np 
import struct
import math

#----------------------------Load data---------------------------------
#----------------------------train_image: list:60000*784---------------
train_images_idx3_ubyte_file = 'C:/Users/tim/Desktop/碩一/碩一下/ML/HW02/train-images-idx3-ubyte'
train_labels_idx1_ubyte_file = 'C:/Users/tim/Desktop/碩一/碩一下/ML/HW02/train-labels-idx1-ubyte'
test_images_idx3_ubyte_file = 'C:/Users/tim/Desktop/碩一/碩一下/ML/HW02/t10k-images-idx3-ubyte'
test_labels_idx1_ubyte_file = 'C:/Users/tim/Desktop/碩一/碩一下/ML/HW02/t10k-labels-idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3通用function
    :param idx3_ubyte_file: idx3 file path
    :return: dataset
    """
    # 讀取2進位data
    bin_data = open(idx3_ubyte_file, 'rb').read()

    #解析head訊息
    offset = 0
    fmt_header = '>iiii'  #read 前4個 unsinged int 32 bit integer
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print ('魔数:%d, 圖片數量: %d张, 圖片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析data
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    #image:60000x(28*28)大小的ndarray
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '張')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((1,num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: dataset
    """
    bin_data = open(idx1_ubyte_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print ('魔數:%d, 圖片數量: %d張' % (magic_number, num_images))

    # 解析data set
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '張')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    -Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    -Only [value] is stored in file, others are only good to understand
    :param idx_ubyte_file: idx文件path
    :return: n*row*col维np.array，n為圖片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)
def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件path
    :return: n*1维np.array，n为圖片數量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    number of image: 10000
    """
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    '''
    number of image label: 10000
    '''
    return decode_idx1_ubyte(idx_ubyte_file)

#-------------------------------------------------------------
#--------------------將data 均存成ndarray --------------------
#-------------------------------------------------------------
"""
    train/test images: (60000/10000,784)
        指令: print(train_images.shape)
    train/test labels: 60000/10000 (一維)
        指令: print(train_labels.shape)
"""
train_images =  load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

#-------------------------------------------------------------
#--------------------算 prior[list]---------------------------
#-------------------------------------------------------------
print("-----------開始算 prior------------")
prior=[0]*10 
num_of_label=[0]*10
for i in range(len(train_labels)):
    prior[int(train_labels[i])]+=1
    num_of_label[int(train_labels[i])]+=1
_sum=0
for i in range(len(prior)):
    prior[i]/=len(train_labels)
prior=np.array(prior)

#-------------------------------------------------------------
#--------------------Setting Mode-----------------------------
#-------------------------------------------------------------
print('-----------Enter the mode setting:-----------')
flag=int(input())


#--------------------In Contunuius mode---------------------------
#--------------------In Contunuius mode---------------------------
#--------------------In Contunuius mode---------------------------
if flag==1:
    print('----------Start of continuous mode:-----------')
#-------------------------------------------------------------
#--------------------算Likelihood (784feature)----------------
#-------------------------------------------------------------
    print('----------開始算 mean & var-----------')
    means = np.zeros((10, 28*28), dtype=np.float64)
    stdevs = np.zeros((10, 28*28), dtype=np.float64)
    for k, y_k in enumerate(np.unique(train_labels)):
        indices = np.where(y_k == train_labels)  #返回index 
        means[k] = np.mean(train_images[indices], axis=0) #return 1*n
        stdevs[k] = np.std(train_images[indices], axis=0)


#-----------------------------------------------------
#--------------------開始算posterior------------------
#-----------------------------------------------------
    '''
        Gaussioan Log Likelihood 
        p(y|Xi) 正比於 連乘p(Xi|y)*prior
    '''
    print('----------開始算posterior----------')
    pred_labels=[0]*10000
    pred_labels=np.array(pred_labels)

    """
        For each 10000 test_images
        Calculate posterior of 0~9 labels of each 10000images
    """
    for im in range(10000): #10000張圖片
        log_likelihood=[0]*10
        log_likelihood=np.array(log_likelihood,dtype=float)
        print('test_image[%d] Posterior (in Log Scale):' %(im))
        for i in range(10): #10個label
            temp=0
            for j in range(784): #784 feature
                if stdevs[i][j]!=0: 
                    log_inside=float(2*(math.pi)*(stdevs[i][j]**2))
                    first_part=float(math.log(log_inside)/2)
                    second_part=float(((test_images[im][j]-means[i][j])**2)/(2*(stdevs[i][j]**2)))
                    temp=float(temp+first_part+second_part)
            temp=temp*math.log(float(prior[i]))
            temp=temp*(-1)
            log_likelihood[i]=float(temp)
        
        """
            Calculate the min of 0~9 posterior of each test_images
            The label with min value is the prediction label
        """
        min_index=0
        _min=float('inf') 
        for s in range(10):
            if _min>log_likelihood[s]:
                _min=log_likelihood[s]
                min_index=s
        
        pred_labels[im]= min_index
    
        """
            Make the sum of each likelihood become 1
            Like the concept of probability
        """
        normalize_sum=0
        for i in range(10):
            normalize_sum+=log_likelihood[i]
        for i in range(10):
            log_likelihood[i]=float(log_likelihood[i]/normalize_sum)
            print('%d : %f' %(i,log_likelihood[i]))
    
        print('Prediction: %d,Ans: %d' %(pred_labels[im],test_labels[im]))
        print('\n')
    
#-----------------------------------------------------
#--------------------開始算error----------------------
#-----------------------------------------------------    
    loss=0
    for i in range(10000):
        if pred_labels[i]!=test_labels[i]:
            loss+=1
    error=loss/10000
    print('error=%f' %(error))




#--------------------In Discrete mode------------------
#--------------------In Discrete mode------------------
#--------------------In Discrete mode------------------
"""
    784features: 不再儲存intensity, 儲存frequency of bins:0~7,8~15,......,248~255
    Create a array 10*784*32 : Store each freq of bin in each features
    In Discrete mode: 
        train_discrete_images即為train_images
"""
if flag==0:
    #train_images=train_images.astype(np.int)
    print("---------Start of Discrete mode-------------")
    
    """
        Calculate freq of bins of train_images
    """
    print('---------Start to calculate freq of bins of train_images---------')
    train_discrete_images=np.zeros((10,784,32),dtype=np.float64)
    for i in range(60000):
        label_num=int(train_labels[i])
        for j in range(784):
            class_of_pixel=int(train_images[i][j]/8)
            train_discrete_images[label_num][j][class_of_pixel]+=1
    #freq: nums/all_nums
    #nums: bin_i / all_nums , i=0~31
    #all_nums: nums_of_imagelabel_i / 60000
    for i in range(10):
        train_discrete_images[i]=train_discrete_images[i]/int(num_of_label[i])
        
    
    print('---------Start to predict---------')
    pred_labels=[0]*10000
    pred_labels=np.array(pred_labels,dtype=float)

    """
        For each 10000 test_images
        Calculate posterior of 0~9 labels of each 10000images
    """
    for im in range(10000): #10000張圖片
        print('test_image[%d] Posterior:' %(im))
        likelihood=[0]*10
        likelihood=np.array(likelihood,dtype=float)
        for i in range(10):
            temp=1
            for j in range(784):
                now_bin=int(test_images[im][j]/8)
                temp*=train_discrete_images[i][j][now_bin]
            temp*=prior[i]
            likelihood[i]=temp
            
        """
            label with max P(X|Y) is the prediction label
            Y:label from 0~9
        """
        max_index=0
        _max=float('-inf')
        for i in range(10):
            if _max<likelihood[i]:
                _max=likelihood[i]
                max_index=i
        pred_labels[im]=max_index
        
        """
            Make the sum of each likelihood become 1
            Like the concept of probability
        """
        normalize=0
        for i in range(10):
            normalize+=likelihood[i]
        for i in range(10):
            likelihood[i]=float(likelihood[i]/normalize)
            print('%d : %f' %(i,likelihood[i]))
            
        """
            show the detail of each test_image with probility of label 0~9
        """    
        print('Prediction: %d , Ans=%d ' %(max_index,test_labels[im]))
        print('\n')
    """
        Error:
    """
    loss=0
    for i in range(10000):
        if pred_labels[i]!=test_labels[i]:
            loss+=1
    error=loss/10000
    print('error: %f' %(error))
            

    
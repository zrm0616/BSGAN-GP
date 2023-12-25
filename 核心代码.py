"""
Dataset of SVHN

仅需将数据集和对应输入数据的函数参数调整；
image_ind = 10
train_data = loadmat('./SVHN/train_32x32.mat')
test_data = loadmat('./SVHN/test_32x32.mat')

TRAIN_X = np.transpose(train_data['X'],(3,0,1,2))
TRAIN_Y = train_data['y']
x_test = np.transpose(test_data['X'],(3,0,1,2))
y_test = test_data['y']

TRAIN_X[TRAIN_Y == 10] = 0
y_test[y_test == 10] = 0
#TRAIN_Y = [num if num != 10 else 0 for num in TRAIN_Y]
TRAIN_X = change_image_shape(TRAIN_X)#将图像按照我们的设置函数标准化

###
# 加载图像，此时是不平衡版本的所有图片
def load_real_samples_B():
    (trainX, trainY) = (TRAIN_X, TRAIN_Y)
    #X = expand_dims(trainX, axis=-1)
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5  # scale from [0,255] to [-1,1] as we will be using tanh activation.
    return [X, trainY]
###

"""
"""
Dataset of Fashion、Mnist
"
3   7    8    9
504	875	3528 891
162	70	468	126
“

"""
import math
import os
import warnings
from keras.utils import to_categorical

warnings.filterwarnings('ignore')  # 出现警告、忽视
from numpy import expand_dims, zeros, ones, asarray
from numpy.random import randn, randint, rand, random
import random
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose
from keras.layers import LeakyReLU, Dropout, Lambda, Activation, ReLU
from matplotlib import pyplot as plt
from keras import backend as K
import numpy as np
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import load_model  # 加载模型函数

# %% --------------------------------------- BAGAN--Fix Seeds -----------------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = tf.keras.initializers.glorot_normal(seed=SEED)

# %% ---------------------------------- 数据准备工作  Data Preparation ---------------------------------------------------------------
# 定义图像大小规模
def change_image_shape(images):
    shape_tuple = images.shape
    if len(shape_tuple) == 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], 1)
    elif shape_tuple == 4 and shape_tuple[-1] > 3:
        images = images.reshape(-1, shape_tuple[-1], shape_tuple[-1], shape_tuple[1])
    return images


# from keras.datasets.mnist import load_data  # 直接定义数据为mnist
from keras.datasets.fashion_mnist import load_data

# 0 T恤（T-shirt/top） 1 裤子（Trouser） 2 套头衫（Pullover） 3 连衣裙（Dress） 4 外套（Coat） 5 凉鞋（Sandal） 6 衬衫（Shirt） 7 运动鞋（Sneaker） 8 包（Bag） 9 靴子（Ankle boot）

(img_our, label_our), (x_test, y_test) = load_data()
print(img_our.shape,label_our.shape)
dict_labels_mnistfashion = {0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt',
                            7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
dict_labels_mnist = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
for c in range(1, 10):  # 创建了一个类别的不平衡、Mnist数据集需要修改数量量
    img_our_B = np.vstack(
        [img_our[label_our != c], img_our[label_our == c][:100 * c]])
    label_our_B = np.append(label_our[label_our != c], np.ones(100 * c) * c)

# 设置通道
channel = label_our.shape[-1]
n_classes = len(np.unique(label_our))
every_labels = np.unique(label_our)
print("类别数组", every_labels)
print("数据集类一共有：", n_classes, "个")
# 超参数优化器初始化
optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
d_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)
g_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.9)


# #------------------------------定义函数---------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 定义独立生成器模型define the standalone generator model
def define_generator(latent_dim):
    in_lat = Input(shape=(latent_dim,))  # 输入latent_dim的长度,shape是读取矩阵的长度，一维数组，长度为latent_dim，即100
    # 从足够密集的节点开始，将其重塑并转化为28x28x1。Start with enough dense nodes to be reshaped and ConvTransposed to 28x28x1
    n_nodes = 256 * 7 * 7  # 节点有12544个

    # 7*7*256
    X = Dense(n_nodes)(in_lat)  # X等于Dense层就是全连接层，用于层方式的初始化的时候。dense帮助神经网络拟合复杂函数用来构建多层感知机，dense数由参数和激活函数组成，
    X = tf.keras.layers.ReLU()(X)  # 激活函数
    # X = LeakyReLU(alpha=0.2)(X)
    X = Reshape((7, 7, 256))(X)  # Reshape将指定的矩阵变换成特定维数7*7*256矩阵（即7个7*256的二维矩阵的三维大矩阵）

    # 14x14x128
    X = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(X)
    X = tf.keras.layers.ReLU()(X)  # 激活函数
    # X = LeakyReLU(alpha=0.2)(X)

    # 28x28x64
    X = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(X)
    X = BatchNormalization()(X)  # 自己添加进去的
    X = tf.keras.layers.ReLU()(X)  # 激活函数
    # X = LeakyReLU(alpha=0.2)(X)

    # 输出
    # 28x28x1
    out_layer = Conv2DTranspose(1, (3, 3), strides=(1, 1), activation='tanh', padding='same')(
        X)  # 反卷积Conv2DTranspose操作，激活函数设为tanh
    # 定义模型
    model = Model(in_lat, out_layer)  # 输入latent_dim的长度，输出
    return model


# 测试生成器输入
gen_model = define_generator(latent_dim=100)  # 生成器latent_dim输入设置为100，表示噪声z向量为100长度
print("输出生成器模型", gen_model.summary())


# -------------------------------------------------------------------------------
# 定义鉴别器
def define_discriminator(in_shape=(28, 28, 1), n_classes=10):
    # def define_discriminator(in_shape=(32,32,3), n_classes=10):
    in_image = Input(shape=in_shape)  # (28,28,1)

    X = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(in_image)  # 卷积核为3步长为2对in_shape=(28,28,1)进行下采样
    X = LeakyReLU(alpha=0.2)(X)
    # X = tf.keras.layers.ReLU()(X)

    X = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(X)  # 卷积层。下采样。卷积核大小3*3，stride步长为2，padding填充值大小为same
    X = LeakyReLU(alpha=0.2)(X)
    # X = tf.keras.layers.ReLU()(X)                                  #leakyRelu激活函数

    X = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(X)  # 卷积层
    X = LeakyReLU(alpha=0.2)(X)
    # X = tf.keras.layers.ReLU()(X)

    X = Flatten()(X)
    X = Dropout(0.4)(X)  # 减少过度拟合minimize overfitting
    X = Dense(n_classes)(X)  # 全连接层

    model = Model(inputs=in_image, outputs=X)

    return model


# -------------------------------------------------------------------------------

def define_sup_discriminator(disc):
    model = Sequential()
    model.add(disc)  # 第一层网络
    model.add(Activation('softmax'))  # 第二层网络
    # Let us use sparse categorical loss so we dont have to convert our Y to categorical
    model.compile(optimizer=d_optimizer,
                  loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    return model


# -------------------------------------------------------------------------------
# 定义无监督分类器
def custom_activation(x):
    Z_x = K.sum(K.exp(x), axis=-1, keepdims=True)
    D_x = Z_x / (Z_x + 1)
    return D_x


def define_unsup_discriminator(disc):
    model = Sequential()  # 建立网络
    model.add(disc)  # 第一层网络
    model.add(Lambda(custom_activation))  # 第二层网络
    model.compile(loss='binary_crossentropy', optimizer=d_optimizer)
    return model

disc = define_discriminator()
disc_sup = define_sup_discriminator(disc)
disc_unsup = define_unsup_discriminator(disc)
print("输出无监督分类器模型", disc.summary())


# -------------------------------------------------------------------------------
def define_gan(gen_model, disc_unsup):
    disc_unsup.trainable = False  # 使unsup。鉴别器不可训练make unsup. discriminator not trainable
    gan_output = disc_unsup(gen_model.output)  # 鉴别器的输入为生成器的输出en. output is the input to disc.
    model = Model(gen_model.input, gan_output)  # 生成器的输入（真实图像)，无监督鉴别器的输出
    model.compile(loss='binary_crossentropy', optimizer=optimizer)  # 判断真伪
    return model


gan_model = define_gan(gen_model, disc_unsup)
print("GAN_MODEL的一些输出层参数", gan_model.summary())

# -------------------------------------------------------------------------------
####################################更换数据集####################################
# 加载图像，此时是平衡所有图片
def load_real_samples(n_classes):
    (trainX, trainY), (_, _) = load_data()
    X = expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    print("所有图像的shape为:", X.shape, trainY.shape)
    return [X, trainY]


# 加载图像，此时是不平衡版本的所有图片
def load_real_samples_B(n_classes=10):
    (trainX, trainY) = (img_our_B, label_our_B)
    X = expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5  #
    return [X, trainY]


# -------------------------------------------------------------------------------
# 选择有监督样本，取部分图片 如5000张，分类设置为10，每类选择100张
def select_supervised_samples(dataset, n_samples=1000, n_classes=10):  # 需自己设定分类n_samples是自己设定的选择多数有监督样本
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)  # Number of amples per class.
    for i in range(n_classes):
        X_with_class = X[y == i]  # get all images for this class
        print('第', i, '类样本的数量', len(X_with_class))
        # X_with_class = X[y.reshape(-1) == i]
        ix = randint(0, len(X_with_class), n_per_class)  # choose random images for each class
        [X_list.append(X_with_class[j]) for j in ix]  # add to list
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(
        y_list)  # Returns a list of 2 numpy arrays corresponding to X and Y


def generate_real_samples(dataset1, n_samples):
    images, labels = dataset1
    ix = randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = ones((n_samples,
              1))
    return [X, labels], y

#generate latent points, to be used as inputs to the generator.
def generate_latent_points(latent_dim, n_samples):
    z_input = randn(latent_dim * n_samples)
    z_input = z_input.reshape(n_samples, latent_dim)  # 重塑输入到网络reshape 是将矩阵
    return z_input



def generate_fake(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)  #
    fake_images = generator.predict(z_input)

    y = zeros((n_samples, 1))
    return fake_images, y


# report accuracy and save plots & the model periodically.
def summarize_performance(step, gen_model, disc_sup, latent_dim, dataset, n_samples=100):
    # 生成假图像样本
    # BAGAN生成显示合成图像代替上面注释掉的这段
    n_classes = len(np.unique(label_our))  # len(np.unique(y_train))就是n_classes
    n = n_classes
    plt.figure(figsize=(2 * n, 2 * (n + 1)))
    np.random.seed(42)  # np.random.seed() 利用随机数种子，指定了一个随机数生成的起始位置，使得每次生成的随机数相同
    x_real = x_test * 0.5 + 0.5

    for i in range(n):  # 每个类便利
        # 第一排显示原始图像
        ax = plt.subplot(n + 1, n, i + 1)
        if channel == 3:  # 如果是彩色图，即通道数为3，就显示图
            plt.imshow(x_real[y_test == i][4])
        else:  # 如果通道数不为3，就显示灰度图
            plt.imshow(x_real[y_test == i][4])
            plt.xlabel(dict_labels_mnist[label_our[i]], fontsize=6)
            plt.gray()  # 灰度函数
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  # 设置坐标轴不可见，以隐藏坐标轴
        # 剩下十排显示生成假图像
        for c in range(n):
            ax = plt.subplot(n_classes + 1, n_classes, (i + 1) * n_classes + 1 + c)
            X, _ = generate_fake(gen_model, latent_dim, n_samples)  # 生成100个假样本展示
            X = (X + 1) / 2.0  # scale to [0,1] for plotting
            if channel == 3:
                plt.imshow(X[i, :, :, 0])
                # plt.imshow(X[i].reshape(28, 28, 1))
            else:
                # plt.imshow(X[y_test==i, :, :, 0])#其中X[, 1:10, :, ]
                # plt.imshow(X[y_test == i].reshape(28, 28, 1))
                plt.imshow(X[i].reshape(28, 28, 1))  # 随机展示类别
                plt.gray()
            ax.get_xaxis().set_visible(False)  # 设置坐标轴不可见，以隐藏坐标轴
            ax.get_yaxis().set_visible(False)
    # 保存图save plot to drive
    filename1 = 'generated_plot_%04d.png' % (step + 1)
    plt.savefig(filename1)
    print('已保存生成%04d图' % (step + 1))  # %04d的意思是保存4位整数
    # plt.show()#绘图后会暂停执行,直到手动关闭当前窗口。plt.pause(time)函数也能实现窗口绘图（不需要plt.show）,但窗口只停留time时间便会自动关闭，然后再继续执行后面代码；
    # 注意plt.pause()会把它之前的所有绘图都绘制在对应坐标系中，而不仅仅是在当前坐标系中绘图；
    plt.pause(5)  ##自己加的，这样就在1秒内自动关闭展示的图
    plt.close()  # 关闭当前显示的图像

    # 计算鉴别器
    X, y = dataset
    _, acc = disc_sup.evaluate(X, y, verbose=0)
    print('鉴别器精度Discriminator Accuracy: %.3f%%' % (acc * 100))
    # 分批保存生成模型save the generator model
    filename2 = 'gen_model_%04d.h5' % (step + 1)
    gen_model.save(filename2)
    print('已保存生成模型%04d .h5' % (step + 1))
    # 保存Discriminator(分类器)模型
    filename3 = 'disc_sup_%04d.h5' % (step + 1)
    disc_sup.save(filename3)
    print('已保存鉴别-监督模型%04d .h5' % (step + 1))
    print('>保存Saved: %s, %s, and %s' % (filename1, filename2, filename3))
    return acc


# ---------------------------------------惩罚梯度----------------------------------------
# 惩罚梯度方法1
def Gradient_Penalty(disc, half_batch, real_images, fake_images):
    """ 计算梯度惩罚Calculates the gradient penalty.
    这种损失是在插值图像上计算的加上鉴别器损失。
    This loss is calculated on an interpolated image and added to the discriminator loss.
    """
    alpha = tf.random.normal([half_batch, 28, 28, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = disc(interpolated)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# ----------------------------损失、精度、训练函数---------------------------------------
# 训练生成器和鉴别器
def train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs, n_batch):
    # 首先选取有标签的样本--10*10个
    X_sup, y_sup = select_supervised_samples(dataset)  # #在原始数据集中平均采样，此时有监督的数据是平均的
    print(X_sup.shape, y_sup.shape)  # 输出样本信息
    # 迭代次数
    bat_per_epo = int(dataset[0].shape[0] / n_batch)  # n_batch
    n_steps = bat_per_epo * n_epochs  # 所有步=迭代次数*周期
    half_batch = int(n_batch / 2)
    print("half_batch全局上的值预计为250，实际为：", half_batch)
    print('设定n_epochs=%d, n_batch=%d, half_batch=%d, b/e=迭代次数%d, 总步长steps=%d ' % (
    n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    #######################  修 改  ####################################################################
    our_acc_epoch = []  # 每一次周期内的训练精度
    our_loss_sup = []
    our_acc_sup = []
    our_loss_d_real = []
    our_loss_d_fake = []
    our_loss_d = []
    our_loss_g = []
    #  枚举周期
    for i in range(n_steps):
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], 250)  # 在1000个带有标签的数据中选择250个，未知真假
        # [Xsup_real, ysup_real], _ = Balance_Real_Real_samples([X_sup, y_sup], half_batch)
        sup_loss, sup_acc = disc_sup.train_on_batch(Xsup_real, ysup_real)
        # D_Loss鉴别器=L真＋L假。L真=L有标记+L无标记

        unlabel_batch = 250
        [X_real, _], y_real = generate_real_samples(dataset, unlabel_batch)  # 选择真实无标记样本，在所有数据中选择2500个作为没有标签的真实样本
        X_fake, y_fake = generate_fake(gen_model, latent_dim, unlabel_batch)  # 训练假# 尝试生成假样本数量为2500
        d_loss_real = disc_unsup.train_on_batch(X_real, y_real)
        d_loss_fake = disc_unsup.train_on_batch(X_fake, y_fake)
        # 添加梯度惩罚----------------------------------------------
        batch_size = n_batch
        # gp = Gradient_Penalty(disc, 900, X_real, X_fake)
        # ## 将梯度惩罚添加到原始鉴别器损失中# 原始d_loss = d_cost + gp * self.gp_weight# 修改后可以用
        # d_loss_fake = d_loss_fake + 0.005 * gp
        # d_loss_real = d_loss_real + 0.005 * gp
        d_loss = d_loss_real + d_loss_fake  # 这个0.1是自己设定的，跟那个weight一样，选一个就行#只用在鉴别器中,将d——loss换成gan_loss试试


        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))  # 生成规模
        gan_loss = gan_model.train_on_batch(X_gan, y_gan)

        print('>%d, [sup损失%.3f,sup准确%.3f], [d真损失%.3f,d假损失%.3f], [d损失%.3f], [g损失%.3f]' % (
            i + 1, sup_loss, sup_acc * 100, d_loss_real, d_loss_fake, d_loss,
            gan_loss))  # .3f的意思是小数点后留3位，显示所有批次的损失值和精确度
        if i % 20 == 0:
            our_loss_sup.append(sup_loss)
            our_acc_sup.append(sup_acc)
            our_loss_d_real.append(d_loss_real)
            our_loss_d_fake.append(d_loss_fake)
            our_loss_d.append(d_loss)
            our_loss_g.append(gan_loss)
        # 定期评估模型性能
        if (i + 1) % (bat_per_epo * 1) == 0:
            print('现在是第%.0f批次，等待计算鉴别器分类精度' % i)
            our_acc_epoch.append(summarize_performance(i, gen_model, disc_sup, latent_dim, dataset))

    ##############################修改绘图#############################################################
    # 绘制训练准确率的折线图
    plt.figure(figsize=(7, 3))
    x2 = np.arange(1, n_epochs + 1, 1)
    plt.plot(x2, our_acc_epoch, label="Every_Epoch_DAcc", color='#B03060')  # 每一周期的分类器精确度
    for a, b in zip(x2, our_acc_epoch):
        plt.text(a, b, round(100 * b, 2), ha='center', va='bottom', fontsize=7)  # 显示每一个点的精确度值,保留2位数
    plt.ylabel('Every_Epoch_DAcc')
    plt.xlabel('Epoch')
    plt.title("3- Every_Epoch_DAcc", fontsize=24)
    plt.legend()
    plt.show()
    print(max(our_acc_epoch))

############################绘图结束############################################
# ---------------------------------- TRAIN训练--------------------------------------
#
latent_dim = 100
disc = define_discriminator()
disc_sup = define_sup_discriminator(disc)
disc_unsup = define_unsup_discriminator(disc)
gen_model = define_generator(latent_dim)
gan_model = define_gan(gen_model, disc_unsup)


# 加载真实样本来定义数据集。
# dataset = load_real_samples(n_classes)  # 调用自建数据函数(输入是一个包含两个numpy数组的列表，X和y)
# 加载不平衡样本来定义数据集
dataset = load_real_samples_B(n_classes)
# 注意:在本例中平衡数据集，1 epoch = 600步。不平衡数据集，1 epoch=549步
#train(gen_model, disc_unsup, disc_sup, gan_model, dataset, latent_dim, n_epochs=25, n_batch=100)
# 一批次100张图，一周期输入600批次。然后迭代10周期==600000万张图



##########################################训练结束################################################################
# Plot generated images
def show_plot(examples, n2):
    if n2 == 0: return print("None")
    n = math.sqrt(n2)
    n = int(n - 0.5)
    print(n, "n2=", n2)
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(np.squeeze(examples[i], axis=-1), cmap='gray')
    plt.pause(5)
    plt.close()
    #plt.show()
def show_plot_fake(examples, n2):
    if n2 == 0: return print("None！！")
    for i in range(10 * 25):
        plt.subplot(10, 25, 1 + i)
        plt.axis('off')
        plt.imshow(np.squeeze(examples[i], axis=-1), cmap='gray')
    plt.pause(5)
    plt.close()
    #plt.show()

netG = load_model('4BSGAN-GP-Fashion h5_B/gen_model_13176.h5')  # Model trained for 100 epochs
netD = load_model('4BSGAN-GP-Fashion h5_B/disc_sup_13176.h5')
"""
latent_points = generate_latent_points(100, 5000)  # Latent dim=100 and n_samples=25
X = netG.predict(latent_points)

y_pred_test = netD.predict(X)  
prediction_test = np.argmax(y_pred_test, axis=1) 
X = (X + 1) / 2.0
X = (X * 255).astype(np.uint8)  
show_plot(X, 100)  
pre_test_label_0 = []
pre_test_label_1 = []
pre_test_label_2 = []
pre_test_label_3 = []
pre_test_label_4 = []
pre_test_label_5 = []
pre_test_label_6 = []
pre_test_label_7 = []
pre_test_label_8 = []
pre_test_label_9 = []
pre_test_img_0 = []
pre_test_img_1 = []
pre_test_img_2 = []
pre_test_img_3 = []
pre_test_img_4 = []
pre_test_img_5 = []
pre_test_img_6 = []
pre_test_img_7 = []
pre_test_img_8 = []
pre_test_img_9 = []
Class_num = 25
X_B_img_fake, X_B_label_fake = [], []
for j in range(10):
    pre_label, pre_img = [], []
    for i in range(len(prediction_test)):
        if prediction_test[i - 1] == j:
            pre_label.append(prediction_test[i - 1])
            pre_img.append(X[i - 1])

    if j == 0:
        len_need = Class_num - len(pre_test_img_0)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_0.append(pre_img[0:len_need])
        pre_test_label_0.append(pre_label[0:len_need])
        for element in pre_img[0:len_need]:
            X_B_img_fake.append(element)
        X_B_label_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
    elif j == 1:
        len_need = Class_num - len(pre_test_img_1)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_1.append(pre_img[0:len_need])
        pre_test_label_1.append(pre_label[0:len_need])
        X_B_img_fake = np.concatenate((X_B_img_fake, pre_img[0:len_need]))
        X_B_label_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
    elif j == 2:
        len_need = Class_num - len(pre_test_img_2)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_2.append(pre_img[0:len_need])
        pre_test_label_2.append(pre_label[0:len_need])
        X_B_img_fake = np.concatenate((X_B_img_fake, pre_img[0:len_need]))
        X_B_label_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
    elif j == 3:
        len_need = Class_num - len(pre_test_img_3)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_3.append(pre_img[0:len_need])
        pre_test_label_3.append(pre_label[0:len_need])
        X_B_img_fake = np.concatenate((X_B_img_fake, pre_img[0:len_need]))
        X_B_label_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
    elif j == 4:
        len_need = Class_num - len(pre_test_img_4)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_4.append(pre_img[0:len_need])
        pre_test_label_4.append(pre_label[0:len_need])
        X_B_img_fake = np.concatenate((X_B_img_fake, pre_img[0:len_need]))
        X_B_label_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
    elif j == 5:
        len_need = Class_num - len(pre_test_img_5)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_5.append(pre_img[0:len_need])
        pre_test_label_5.append(pre_label[0:len_need])
        X_B_img_fake = np.concatenate((X_B_img_fake, pre_img[0:len_need]))
        X_B_label_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
    elif j == 6:
        len_need = Class_num - len(pre_test_img_6)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_6.append(pre_img[0:len_need])
        pre_test_label_6.append(pre_label[0:len_need])
        X_B_img_fake = np.concatenate((X_B_img_fake, pre_img[0:len_need]))
        X_B_label_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
    elif j == 7:
        len_need = Class_num - len(pre_test_img_7)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_7.append(pre_img[0:len_need])
        pre_test_label_7.append(pre_label[0:len_need])
        X_B_img_fake = np.concatenate((X_B_img_fake, pre_img[0:len_need]))
        X_B_label_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
    elif j == 8:
        len_need = Class_num - len(pre_test_img_8)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_8.append(pre_img[0:len_need])
        pre_test_label_8.append(pre_label[0:len_need])
        X_B_img_fake = np.concatenate((X_B_img_fake, pre_img[0:len_need]))
        X_B_label_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
    elif j == 9:
        len_need = Class_num - len(pre_test_img_9)  
        if len(pre_img) < len_need:
            len_need = len(pre_img)
        pre_test_img_9.append(pre_img[0:len_need])
        pre_test_label_9.append(pre_label[0:len_need])
        X_B_img_fake = np.concatenate((X_B_img_fake, pre_img[0:len_need]))
        X_B_labexl_fake = np.concatenate((X_B_label_fake, pre_label[0:len_need]))
show_plot_fake(X_B_img_fake, len(X_B_img_fake))
X_sup, y_sup = select_supervised_samples(dataset) 
[Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], 250)  
sup_loss, sup_acc = netD.train_on_batch(Xsup_real, ysup_real)
print("有监督netD的精度：", sup_acc)
unlabel_batch = 250
[X_real, _], y_real = generate_real_samples(dataset, unlabel_batch)  
y_fake = zeros((250, 1))
X_fake = X_B_img_fake  
d_loss_real = disc_unsup.train_on_batch(X_real, y_real)
d_loss_fake = disc_unsup.train_on_batch(X_fake, y_fake)
d_loss = d_loss_real + d_loss_fake  
# X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))  
# gan_loss = gan_model.train_on_batch(X_gan, y_gan)

Acc_NetD = summarize_performance(13176, netG, netD, latent_dim, dataset)
print("最后鉴别器的总分类精度", Acc_NetD)
"""
# -----------------------------------------分析评估-----------------------------------------------------------------
# EVALUATE THE SUPERVISED DISCRIMINATOR ON TEST DATA
# This is the model we want as a classifier.

disc_sup_trained_model = load_model('4BSGAN-GP-Fashion h5_B/disc_sup_13177.h5')
(_, _), (testX, testy) = load_data()
for c in range(1, 10):  # 创建了一个类别的不平衡
    img_our_B_test = np.vstack(
        [testX[testy != c], testX[testy == c][:7 * c]])
    label_our_B_test = np.append(testy[testy != c], np.ones(7 * c) * c)
(testX, testy) = (img_our_B_test, label_our_B_test)
# expand to 3d, e.g. add channels
testX = expand_dims(testX, axis=-1)
#convert from ints to floats
testX = testX.astype('float32')

# scale from [0,255] to [-1,1]
testX = (testX - 127.5) / 127.5
# evaluate the model
_, test_acc = disc_sup_trained_model.evaluate(testX, testy, verbose=0)
print('Test Accuracy: %.3f%%' % (test_acc * 100))

# Predicting the Test set results
y_pred_test = disc_sup_trained_model.predict(testX)
prediction_test = np.argmax(y_pred_test, axis=1)  # 预测类别，返回每行或每列的最大值所在下标索引
print("", prediction_test)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(6, 6))
cm = confusion_matrix(testy, prediction_test)  # 混淆矩阵。矩阵的每一行代表实际的类别，而每一列代表预测的类别。
print("cm-------------------", cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
# plt.title('Confusion matrix for:\n{}'.format('BSGAN-GP'))
plt.title('Confusion matrix')
plt.show()

# -------------------------------------------------------
test_labels = to_categorical(testy)
predictions = disc_sup_trained_model.predict(testX)
predictionsnum = cm.diagonal()
Acc_i = []
for i in range(10):
    true_count = sum(test_labels[:, i] == 1)
    pred_count = predictionsnum[i]
    Acc_i.append(pred_count / true_count)
    print(f"Class {i}: Accuracy = {Acc_i[-1]:.3%}")


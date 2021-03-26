# deep learning project
# sequence-based binary classification
#TODO: push to GitHub

""" requirements
每个功能块必须加注释
必须设定“__main__”函数,不能直接在空py文件上开始写！
独立功能单元必须放到独立的包里，在主程序中通过“import”导入
变量与命名规范
对于类：类名首字母必须大写！构造函数与析构函数必须写清楚！类的方法采用全小写标志！类内私有变量一律以下划线开头！
对于变量：一切变量采用驼峰命名法
对于方法：方法本体在一行或者三行以内的一律使用匿名函数
"""

#%% import modules
from model_lstm import *


# %% read files
# X_onehot will be (m, Tx, num_classes), Y will be 0 or 1
X, ids = read_files('../type 2 diabetes Raw Data/all-seq')
Y, _ = read_files('../type 2 diabetes Raw Data/all-ids-phe')
Y = np.array(Y,dtype='float64')
X_onehot = seq_to_one_hot(X)
# TODO: This step cost much time! Need to improve!
# solution: save preprocessed data
np.savez('XY.npz', X_onehot, Y)

#%% load processed data to save time
data = np.load('../Deep Learning Project/PythonVersion/XY.npz')
X_onehot = data['arr_0']
Y = data['arr_1']

#%% divide sets
X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_train_test(X_onehot, Y, proportion=(0.8,0.1,0.1))

#%% RNN model and training, not using now
# the paper:
# a (1-4-100-100-2048-2) architecture where 4-dimensional embeddings are passed through a
# BLSTM, attention and fully connected layer
# with 100, 100 and 2048 units respectively.
# Values for batch size, dropout, recurrent dropout, and epochs
# are tuned to 128, 0.5, 0.2 and 50 respectively.

# model= RNN_model(Tx=X_train.shape[1], n_a=100, seq_classes = 2)


#%% LSTM model
# initialize. constants are imported from model_lstm
m = X_train.shape[0]
s0 = np.zeros((m, ns))

encoder = model2_encoder(X_train.shape)
decoder = model2_decoder((m, latent_units), Tx)
model = encoder
"""TODO:
To train the autoencoder, we have to merge the two models; but to train the classifier, we need to
merge the encoder and the simple RNN. It means that I had better write a class, three separate models
and two merged models.
Plus, I'm not sure how the weights are extracted and applied to another model.
Moreover, a lot of hyper parameters are not determined:
- the decoder (na, ns is just ok..)
- the classifier and model 2a
and integrated gradient paper?
"""

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([X_train, s0], Y_train, epochs=100, batch_size=100)


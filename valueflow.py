# -*- coding: utf-8 -*-
import numpy as np

def tanh_forward(X, W, b):
    """
    Inputs:
    -X:(m,idim)维输入数据
    -W:(idim,odim)维数据
    -b:(odim,)维数据
    Returns:
    -out:(m,odim)维输出数据
    """
    z = X.dot(W) + b
    out = np.tanh(z)
    return out

def tanh_backward(dy, X, W, b, reg_lambda=1e-4):
    """
    Inputs：
    -dy:(m,odim)维数据，dl/dy的值
    -X:(m,idim)维数据，输入batch
    -W:(idim,odim)维数据
    -b:(odim,)维数据
    -reg_lambda:float 正则化参数
    Returns:
    -dw:(idim,odim)维数据，损失函数对w的梯度
    -db:(odim,)维数据，对b的梯度
    -dX:(m,idim)维数据，对X的梯度，等于前一层的dy
    """
    z = X.dot(W) + b # (m,odim)
    dz = dy * (1-np.power(np.tanh(z),2)) # (m,odim)
    dw = np.dot(X.T, dz) # (idim, odim)
    db = np.sum(dz, axis=0)
    dw += reg_lambda * W
    
    dX = dz.dot(W.T)
    return (dw, db, dX)

def softmax_forward(X, W, b):
    """
    Inputs:
    -X:(m,idim)维输入数据
    -W:(idim,odim)维数据
    -b:(odim,)维数据
    Returns:
    -out:(m,odim)维输出数据
    """
    z = X.dot(W) + b
    exp_scores = np.exp(z)
    out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return out

def softmax_backward(dz, X, W, b, reg_lambda=1e-4):
    """
    Inputs：
    -dz:(m,odim)维数据，dl/dz的值 cross_entropy的dz是prob-y
    -X:(m,idim)维数据，输入batch
    -W:(idim,odim)维数据
    -b:(odim,)维数据
    -reg_lambda:float 正则化参数
    Returns:
    -dw:(idim,odim)维数据，损失函数对w的梯度
    -db:(odim,)维数据，对b的梯度
    -dX:(m,idim)维数据，对X的梯度，等于前一层的dy
    """
    dw = np.dot(X.T, dz) # (idim, odim)
    db = np.sum(dz, axis=0)
    dw += reg_lambda * W
    
    dX = dz.dot(W.T)
    return (dw, db, dX)

def cross_entropy(Y, prob, encoded=False):
    """
    Inputs:
    -Y:(m,odim)真实标签
    -prob:(m,odim)softmax的结果
    Returns:
    -loss: float 损失值
    -dz: (m,odim) dl/dz = prob-y
    """
    if not encoded:
        catagories = list(np.unique(Y))
        labels = np.zeros((Y.shape[0], len(catagories)), dtype=np.float64)
        for i in range(labels.shape[0]):
            labels[i, catagories.index(Y[i])] = 1
    else:
        labels = Y
        
    loss = -np.mean(np.sum(labels * np.log(prob), axis=1))
    dz = prob - labels # (m,odim)
    
    return (loss, dz)


def relu_forward(X, W, b): 
    """
    Inputs:
    -X:(m,idim)维输入数据
    -W:(idim,odim)维数据
    -b:(odim,)维数据
    Returns:
    -out:(m,odim)维输出数据
    """
    z = X.dot(W) + b
    out = np.where(z>=0, z, 0)
    return out

def relu_backward(dy, X, W, b, reg_lambda=1e-4):
    """
    Inputs：
    -dy:(m,odim)维数据，dl/dy的值
    -X:(m,idim)维数据，输入batch
    -W:(idim,odim)维数据
    -b:(odim,)维数据
    -reg_lambda:float 正则化参数
    Returns:
    -dw:(idim,odim)维数据，损失函数对w的梯度
    -db:(odim,)维数据，对b的梯度
    -dX:(m,idim)维数据，对X的梯度，等于前一层的dy
    """
    z = X.dot(W) + b # (m,odim)
    dz = np.where(z>=0, dy, 0) # (m, odim)
    
    dw = np.dot(X.T, dz) # (idim, odim)
    db = np.sum(dz, axis=0)
    dw += reg_lambda * W
    
    dX = dz.dot(W.T)
    return (dw, db, dX)

def BN_forward(X, gamma, beta, mode='train', epsilon=1e-5, momentum=0.9, running_mean=None, running_var=None):
    """
    Inputs:
    -X:(m,odim)
    -gamma:(odim,) 要学习的参数
    -beta:(odim,) 要学习的参数
    -mode:'train'或者'test'，对应训练情况和测试情况
    -epsilon:公式里的修正值 1e-5
    -momentum:用于计算训练过程的均值方差期望值 0.9
    -running_mean:(odim,)存储训练集的动态均值
    -running_var:(odim,)存储训练集的动态方差
    Returns:
    -out:(m,odim)
    -cache:存储反向传播所需数据
    """
    m, odim = X.shape
    if type(running_mean) != np.ndarray:
        running_mean = np.zeros(odim, dtype=X.dtype)
    if type(running_var) != np.ndarray:
        running_var = np.zeros(odim, dtype=X.dtype)
    
    out, cache = None, None
    
    if mode == 'train':
        sample_mean = np.mean(X, axis=0) # (1,odim)
        sample_var = np.var(X, axis=0) # (1,odim)
        x_hat = (X - sample_mean) / np.sqrt(sample_var + epsilon) # (m,odim)
        out = gamma * x_hat + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = {'X':X, 'gamma':gamma, 'beta':beta, 'x_hat':x_hat,
                 'epsilon':epsilon,'momentum':momentum, 
                 'running_mean':running_mean,'running_var':running_var, 
                 'sample_mean':sample_mean,'sample_var':sample_var}
    elif mode=='test':
        x_hat = (X - running_mean) / np.sqrt(running_var + epsilon)
        out = gamma * x_hat + beta
    else:
        raise ValueError("请输入正确的模式: train / test")
        
    return out, cache

def BN_backward(dy, cache):
    """
    Inputs:
    -dy:(m,odim) dl/dy
    -cache:所需的中间变量
    Returns:
    -dX:(m,idim)维数据，对X的梯度，等于前一层的dy
    -dgamma:(idim,)对gamma的梯度
    -dbeta:(idim,)对beta的梯度
    """
    assert isinstance(cache, dict)
    X, gamma, beta, x_hat, sample_mean, sample_var, epsilon = \
    cache['X'], cache['gamma'], cache['beta'], cache['x_hat'], \
    cache['sample_mean'], cache['sample_var'], cache['epsilon']
    
    m = X.shape[0]
    
    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)
    
    dx_hat = dy * gamma
    dsigma = -0.5 * np.sum(dx_hat * (X - sample_mean), axis=0) * \
            np.power(sample_var + epsilon, -1.5)
    dmu = -np.sum(dx_hat / np.sqrt(sample_var + epsilon), axis=0) - \
        2 * dsigma*np.sum(X-sample_mean, axis=0) / m
    dX = dx_hat / np.sqrt(sample_var + epsilon) + 2.0 * dsigma * (X - sample_mean) / \
        m + dmu / m
    
    return dX, dgamma, dbeta

class Dataset():
    """
    输入文件列表，建立数据集类，包括shuffle私有方法，用于数据读完后打乱数据，私有变量为数据的长度，开始的下标，数据，标签
    """
    def __init__(self, data, labels, need_shuffle=False):
        self._data = data
        self._labels = labels
        print(self._data.shape)
        print(self._labels.shape)
        self._need_shuffle = need_shuffle
        self._data_num = self._data.shape[0]
        self._indicator = 0
    
    def next_batch(self,batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._data_num:
            if self._need_shuffle:
                self._shuffle()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("There is no sample in dataset.")
        if end_indicator > self._data_num:
            raise Exception("data_batch is larger than data size")
        data_batch, labels_batch = (self._data[self._indicator:end_indicator],\
                                    self._labels[self._indicator:end_indicator])
        self._indicator = end_indicator
        return (data_batch, labels_batch)
    
    def _shuffle(self):
        index = np.random.permutation(self._data_num)
        self._data = self._data[index]
        self._labels = self._labels[index]
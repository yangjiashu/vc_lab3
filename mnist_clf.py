import valueflow as vf
import numpy as np
import sys
sys.path.append('mnist')
import mnist
import matplotlib.pyplot as plt

HIDDEN_UNIT_1 = 50
HIDDEN_UNIT_2 = 50
HIDDEN_UNIT_3 = 20
EPOCHS = 10000
BATCH_SIZE = 1000
LR = 6e-6
LAMBDA = 1e-4

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()

n_train, w, h = train_images.shape
X_train = train_images.reshape( (n_train, w*h) ).astype(np.float64)
Y_train = train_labels

n_test, w, h = test_images.shape
X_test = test_images.reshape( (n_test, w*h) ).astype(np.float64)
Y_test = test_labels


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

def weight_b(m, n, dtype='w'):
    if dtype=='w':
        return np.random.randn(m,n).astype(np.float64) / np.sqrt(m/2).astype(np.float64)
    elif dtype=='b':
        return np.zeros((m,n), dtype=np.float64)
    else:
        raise ValueError('请输入正确的值：w / b')

def accuracy(x,model,labels,act='relu'):
    
    hidden1 = vf.tanh_forward(x, model['w1'], model['b1'])
    if act=='relu':
        hidden2 = vf.relu_forward(hidden1, model['w2'], model['b2'])
        hidden3 = vf.relu_forward(hidden2, model['w3'], model['b3'])
    elif act=='tanh':
        hidden2 = vf.tanh_forward(hidden1, model['w2'], model['b2'])
        hidden3 = vf.tanh_forward(hidden2, model['w3'], model['b3'])
    probs = vf.softmax_forward(hidden3, model['w4'], model['b4'])
    
    y_hat = np.argmax(probs, axis=1)
    correct_prediction = np.equal(y_hat, labels).astype(np.float32)
    accuracy = np.mean(correct_prediction)
    
    return accuracy
    
def train(model, epochs, batch_size, lr, act='relu'):
    
    train_losses = []
    test_losses = []
    axis = []
    
    # forward
    dataset = vf.Dataset(X_train, Y_train, True)
    for i in range(epochs):
        input_data, labels = dataset.next_batch(batch_size)    
        
        # forward
        hidden1 = vf.tanh_forward(input_data, model['w1'], model['b1'])
        if act=='relu':
            hidden2 = vf.relu_forward(hidden1, model['w2'], model['b2'])
            hidden3 = vf.relu_forward(hidden2, model['w3'], model['b3'])
        elif act=='tanh':
            hidden2 = vf.tanh_forward(hidden1, model['w2'], model['b2'])
            hidden3 = vf.tanh_forward(hidden2, model['w3'], model['b3'])
        else:
            raise ValueError('请输入正确的激活函数: relu / tanh')
        probs = vf.softmax_forward(hidden3, model['w4'], model['b4'])
        
        # loss
        train_loss, dz = vf.cross_entropy(labels, probs)
        
        # backward
        dw4, db4, dflow3 = vf.softmax_backward(dz, hidden3, model['w4'], model['b4'], reg_lambda=LAMBDA)
        if act=='relu':
            dw3, db3, dflow2 = vf.relu_backward(dflow3, hidden2, model['w3'], model['b3'], reg_lambda=LAMBDA)
            dw2, db2, dflow1 = vf.relu_backward(dflow2, hidden1, model['w2'], model['b2'], reg_lambda=LAMBDA)
        elif act=='tanh':
            dw3, db3, dflow2 = vf.tanh_backward(dflow3, hidden2, model['w3'], model['b3'], reg_lambda=LAMBDA)
            dw2, db2, dflow1 = vf.tanh_backward(dflow2, hidden1, model['w2'], model['b2'], reg_lambda=LAMBDA)
        dw1, db1, _ = vf.tanh_backward(dflow1, input_data, model['w1'], model['b1'], reg_lambda=LAMBDA)
        
        # update
        model['w4'] += -lr * dw4
        model['b4'] += -lr * db4
        model['w3'] += -lr * dw3
        model['b3'] += -lr * db3
        model['w2'] += -lr * dw2
        model['b2'] += -lr * db2
        model['w1'] += -lr * dw1
        model['b1'] += -lr * db1
        
        if (i + 1) % 10 == 0:
            hidden1 = vf.tanh_forward(X_test, model['w1'], model['b1'])
            if act=='relu':
                hidden2 = vf.relu_forward(hidden1, model['w2'], model['b2'])
                hidden3 = vf.relu_forward(hidden2, model['w3'], model['b3'])
            elif act=='tanh':
                hidden2 = vf.tanh_forward(hidden1, model['w2'], model['b2'])
                hidden3 = vf.tanh_forward(hidden2, model['w3'], model['b3'])
            probs = vf.softmax_forward(hidden3, model['w4'], model['b4'])
            
            # loss
            test_loss, dz = vf.cross_entropy(Y_test, probs)
            
            if (i + 1) % 50 == 0:
                if act=='relu':
                    acc = accuracy(X_test, model, Y_test, act='relu')
                elif act=='tanh':
                    acc = accuracy(X_test, model, Y_test, act='tanh')
                print('step %d: loss: %.4f, acc: %.4f' % (i+1, test_loss, acc))
            else:
                print('step %d: loss: %.4f' % (i+1, test_loss))
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            axis.append(i+1)
    return (train_losses, test_losses, axis)

def main():
    # initialize  
    model = {}
    model['w1'] = weight_b(784, HIDDEN_UNIT_1, 'w')
    model['b1'] = weight_b(1, HIDDEN_UNIT_1, 'b')
    model['w2'] = weight_b(HIDDEN_UNIT_1, HIDDEN_UNIT_2, 'w')
    model['b2'] = weight_b(1, HIDDEN_UNIT_2, 'b')
    model['w3'] = weight_b(HIDDEN_UNIT_2, HIDDEN_UNIT_3, 'w')
    model['b3'] = weight_b(1, HIDDEN_UNIT_3, 'b')
    model['w4'] = weight_b(HIDDEN_UNIT_3, 10, 'w')
    model['b4'] = weight_b(1, 10, 'b')

    
    train_losses, test_losses, axis = train(model, EPOCHS, BATCH_SIZE, LR, act='tanh')
    plt.figure()
    plt.subplot(121)
    plt.title('tanh')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0.1,0.6)
    plt.plot(axis, train_losses, c='blue', label='train')   
    plt.plot(axis, test_losses, c='red', label='test')
    plt.legend()
    
    train_losses, test_losses, axis = train(model, EPOCHS, BATCH_SIZE, LR, act='relu')
    plt.subplot(122)
    plt.title('relu')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0.1,0.6)
    plt.plot(axis, train_losses, c='blue', label='train')
    plt.plot(axis, test_losses, c='red', label='test')
    plt.legend()
    
    plt.show()
    
if __name__ == "__main__":
    main()
    

    
    
    
    
    
    
    
    
import valueflow as vf
import numpy as np
import sys
sys.path.append('mnist')
import mnist

HIDDEN_UNIT_1 = 50
HIDDEN_UNIT_2 = 50
HIDDEN_UNIT_3 = 20
EPOCHS = 1000
BATCH_SIZE = 6000
LR = 0.00001
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

def accuracy(x,model,labels):
    w1 = model['w1']
    b1 = model['b1']
    w2 = model['w2']
    b2 = model['b2']
    w3 = model['w3']
    b3 = model['b3']
    w4 = model['w4']
    b4 = model['b4']
    
    hidden1 = vf.tanh_forward(x, w1, b1)
    hidden2 = vf.relu_forward(hidden1, w2, b2)
    hidden3 = vf.relu_forward(hidden2, w3, b3)
    probs = vf.softmax_forward(hidden3, w4, b4)
    y_hat = np.argmax(probs, axis=1)
    correct_prediction = np.equal(y_hat, labels).astype(np.float32)
    accuracy = np.mean(correct_prediction)
    
    return accuracy
    
def train(model, epochs, batch_size, lr):
    w1 = model['w1']
    b1 = model['b1']
    w2 = model['w2']
    b2 = model['b2']
    w3 = model['w3']
    b3 = model['b3']
    w4 = model['w4']
    b4 = model['b4']
    # forward
    dataset = vf.Dataset(X_train, Y_train, True)
    for i in range(epochs):
        input_data, labels = dataset.next_batch(batch_size)    
        
        # forward
        hidden1 = vf.tanh_forward(input_data, w1, b1)
        hidden2 = vf.relu_forward(hidden1, w2, b2)
        hidden3 = vf.relu_forward(hidden2, w3, b3)
        probs = vf.softmax_forward(hidden3, w4, b4)
        
        # loss
        loss, dz = vf.cross_entropy(labels, probs)
        # backward
        dw4, db4, dflow3 = vf.softmax_backward(dz, hidden3, w4, b4, reg_lambda=LAMBDA)
        dw3, db3, dflow2 = vf.relu_backward(dflow3, hidden2, w3, b3, reg_lambda=LAMBDA)
        dw2, db2, dflow1 = vf.relu_backward(dflow2, hidden1, w2, b2, reg_lambda=LAMBDA)
        dw1, db1, _ = vf.tanh_backward(dflow1, input_data, w1, b1, reg_lambda=LAMBDA)
        
        # update
        w4 += -lr * dw4
        b4 += -lr * db4
        w3 += -lr * dw3
        b3 += -lr * db3
        w2 += -lr * dw2
        b2 += -lr * db2
        w1 += -lr * dw1
        b1 += -lr * db1
        print(loss)
    model['w4'] = w4
    model['b4'] = b4
    model['w3'] = w3
    model['b3'] = b3
    model['w2'] = w2
    model['b2'] = b2
    model['w1'] = w1
    model['b1'] = b1
    

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

    
    train(model, EPOCHS, BATCH_SIZE, LR)
    
    print(accuracy(X_train, model, Y_train))
if __name__ == "__main__":
    main()
    

    
    
    
    
    
    
    
    
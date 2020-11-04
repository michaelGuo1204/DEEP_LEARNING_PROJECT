# utilizable functions
# after changing, we have to rerun it and rerun "import" in main
import numpy as np
from keras.utils import to_categorical


# %% data processing
def read_files(filename):
    data = []
    ids = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.split('\t')  # id and seq or label was separated by Tab
            id = line[0]
            ids.append(id)
            datum = np.array(
                line[1].rstrip().strip(' ').split(' '))  # while nucleotides were separated by space. also remove /n
            data.append(datum)

    return np.array(data), ids  # return X and Y as numpy array


seq2one_hot = {'0': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4}


def seq_to_one_hot(X):  # input size should be (m, Tx)
    num_classes = 5  # the 5th number is left for N in the paper, namely a random nucleotide
    X_onehot = []  # a list of (Tx,5) arrays
    for x in X:
        x = [seq2one_hot[i] for i in x]
        x_onehot = np.zeros((len(x), num_classes))
        for idx in range(len(x)):
            x_onehot[idx,:] = to_categorical(x[idx], num_classes=num_classes)

        X_onehot.append(x_onehot)

    return np.array(X_onehot)


def divide_train_test(X_all, Y_all, proportion=(0.7,0.2,0.1)):
    train, val, test = proportion
    np.random.seed(0)  # fixed random numbers

    m, T_x, num_classes = X_all.shape
    idx = np.random.rand(m)
    train_idx = [i for i in range(m) if idx[i] < train]
    test_idx = [i for i in range(m) if idx[i] > 1 - test]
    val_idx = list(set(range(m)).difference(set(train_idx)).difference(set(test_idx)))

    X_train = X_all[train_idx, :]
    Y_train = Y_all[train_idx, :]
    X_val = X_all[val_idx, :]
    Y_val = Y_all[val_idx, :]
    X_test = X_all[test_idx, :]
    Y_test = Y_all[test_idx, :]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


#%% model1: RNN. not using
# def RNN_model(Tx, n_a, seq_classes = 2):
#     """
#     Implement the model
#
#     Arguments:
#     Tx -- length of the sequence
#     n_a -- the number of activations used in our model
#     num_classes -- number of unique values
#
#     Returns:
#     model -- a keras model
#     """
#
#     # Define the input of your model with a shape
#     X = Input(shape=(Tx, seq_classes))
#
#     # Define s0, initial hidden state for the decoder LSTM
#     a0 = Input(shape=(n_a,), name='a0')
#     c0 = Input(shape=(n_a,), name='c0')
#     a = a0
#     c = c0
#
#     ### START CODE HERE ###
#     # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
#     outputs = []
#
#     # Step 2: Loop
#     for t in range(Tx):
#         # Step 2.A: select the "t"th time step vector from X.
#         x = Lambda(lambda x: X[:, t, :])(X)
#         # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
#         x = reshapor(x)
#         # Step 2.C: Perform one step of the LSTM_cell
#         a, _, c = LSTM_cell(x, initial_state=[a, c])
#         # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
#         out = densor(a)
#         # Step 2.E: add the output to "outputs"
#         outputs.append(out)
#
#     # Step 3: Create model instance
#     model = Model(inputs=[X, a0, c0], outputs=outputs)
#
#     return model


#%% main
# def main():
#     print("1")
#     return 0
#
#
# if __name__ == "__main__":
#     main()

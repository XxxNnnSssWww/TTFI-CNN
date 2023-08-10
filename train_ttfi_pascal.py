import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import math
import os
from model.ecanet import build_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def get_row_sum(matrix, num_col, num_row):
    sum = 0
    for ii in range(num_col):
        sum += matrix[num_row][ii]
    return sum
def get_col_sum(matrix, num_row, num_col):
    sum = 0
    for ii in range(num_row):
        sum += matrix[ii][num_col]
    return sum

def pr(con_mat):
    pr0 = con_mat[0, 0] / (get_col_sum(con_mat, 3, 0))
    pr1 = con_mat[1, 1] / (get_col_sum(con_mat, 3, 1))
    pr2 = con_mat[2, 2] / (get_col_sum(con_mat, 3, 2))
    return pr0, pr1, pr2

def se(con_mat):
    se0 = con_mat[0, 0] / (get_row_sum(con_mat, 4, 0))
    se1 = con_mat[1, 1] / (get_row_sum(con_mat, 4, 1))
    se2 = con_mat[2, 2] / (get_row_sum(con_mat, 4, 2))
    se3 = con_mat[3, 3] / (get_row_sum(con_mat, 4, 3))
    return se0, se1, se2, se3


def predict(prediction):
    ans = []
    for i in range(len(prediction)):
        pred = np.argmax(prediction[i])
        ans.append(pred)
    return ans


def main():
    np.random.seed(0)
    parser = argparse.ArgumentParser(description="argument of 2dcnn net")
    parser.add_argument('--save_path', type=str, default='result/txt/pascalB1.txt', help='path of saved result')
    parser.add_argument('--save_model', type=str, default='result/model/pascal_B_kmeans_{}.h5', help='path of saved model result')
    parser.add_argument('--data_path', type=str, default='data/PASCAL/B1', help='path of load data')

    parser.add_argument('--train_data1_path', type=str, default='train_data1.npy', help='path of load data')
    parser.add_argument('--train_data2_path', type=str, default='train_data2.npy', help='path of load data')
    parser.add_argument('--train_label_path', type=str, default='train_label.npy', help='path of load label')
    parser.add_argument('--val_data1_path', type=str, default='val_data1.npy', help='path of load data')
    parser.add_argument('--val_data2_path', type=str, default='val_data2.npy', help='path of load data')
    parser.add_argument('--val_label_path', type=str, default='val_label.npy', help='path of load label')
    parser.add_argument('--test_data1_path', type=str, default='test_data1.npy', help='path of load data')
    parser.add_argument('--test_data2_path', type=str, default='test_data2.npy', help='path of load data')
    parser.add_argument('--test_label_path', type=str, default='test_label.npy', help='path of load label')

    parser.add_argument('--high', type=int, default=149, help='high of 2d feature')
    parser.add_argument('--wide', type=int, default=39, help='high of 2d feature')
    parser.add_argument('--num_epochs', type=int, default=100, help='epuch')
    parser.add_argument('--num_class', type=int, default=3, help='number of class')
    parser.add_argument('--fold', type=int, default=5, help='fold')
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--length', type=int, default=3000)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('begin load train_data1...')
    train_data1 = np.load(os.path.join(args.data_path, args.train_data1_path), mmap_mode='r')
    print('begin load train_data2...')
    train_data2 = np.load(os.path.join(args.data_path, args.train_data2_path), mmap_mode='r')
    print('load train label!')
    train_label = np.load(os.path.join(args.data_path, args.train_label_path), mmap_mode='r')

    print('load val data1!')
    val_data1 = np.load(os.path.join(args.data_path, args.val_data1_path), mmap_mode='r')
    print('load val data2!')
    val_data2 = np.load(os.path.join(args.data_path, args.val_data2_path), mmap_mode='r')
    print('load val label')
    val_label = np.load(os.path.join(args.data_path, args.val_label_path), mmap_mode='r')

    print('load test data1')
    test_data1 = np.load(os.path.join(args.data_path, args.test_data1_path))
    print('load test data2')
    test_data2 = np.load(os.path.join(args.data_path, args.test_data2_path))
    print('load test label')
    test_label = np.load(os.path.join(args.data_path, args.test_label_path))
    print("begin train...")

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='TB_log/pascal_B2')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                      cooldown=0,
                                                      patience=10,
                                                      min_lr=0.5e-6)
    my_callbacks = [lr_reducer, tensorboard]

    for j in range(args.fold):
        print("this is fold:", j)
        model = build_model(args.num_class, args.length, args.high, args.wide)
        print('begin train')

        model.fit([train_data1, train_data2], train_label,
                  epochs=args.num_epochs,
                  batch_size=32,
                  #class_weight=weigh,
                  validation_data=([val_data1, val_data2], val_label),
                  verbose=1,
                  callbacks=my_callbacks
                  )
        test_predictions = model.predict([test_data1, test_data2])
        test_predictions = predict(test_predictions)
        con_mat = confusion_matrix(test_label, test_predictions)
        pr0, pr1, pr2 = pr(con_mat)
        file = open(args.save_path, "a")
        file.write(str(j) + ": " + str(pr0) + " " + str(pr1) + " " + str(pr2) + " " + '\n')
        file.write(str(con_mat) + '\n')
        file.close()
        print("result save success!")
        model.save(args.save_model.format(j))
        print("save success")
    print("model train success!")
    '''
    model.save(args.save_path)
    print("save success")
    print('load model...')
    model = keras.models.load_model(args.save_path)
    print('load model success!')
    loss, acc = model.evaluate(test_data, test_label)
    print(acc)
    '''

if __name__ == '__main__':
    main()
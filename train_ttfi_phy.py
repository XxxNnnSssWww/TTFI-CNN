import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import math
import os
from model.ecanet import build_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def acc_sp_se_macc_pr_f1(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    tp = con_mat[1, 1]
    fn = con_mat[1, 0]
    fp = con_mat[0, 1]
    tn = con_mat[0, 0]
    acc = (tp + tn) / (tp+fp+tn+fn)
    spe = tn / (tn + fp)
    se = tp / (tp + fn)
    macc = (spe + se) / 2
    pr = tp / (tp + fp)
    f1 = 2*(pr * se) / (pr + se)
    return acc, spe, se, macc, pr, f1

def predict(prediction):
    ans = []
    for i in range(len(prediction)):
        pred = np.argmax(prediction[i])
        ans.append(pred)
    return ans


def main():
    np.random.seed(0)
    '''
    rn.seed(0)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                  inter_op_parallelism_threads=1)
    tf.compat.v1.set_random_seed(0)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    '''
    parser = argparse.ArgumentParser(description="argument of 2dcnn net")
    parser.add_argument('--save_path', type=str, default='result/txt/mfcc.txt', help='path of saved result')
    parser.add_argument('--save_model', type=str, default='result/model/phy_{}.h5', help='path of saved model result')
    parser.add_argument('--data_path', type=str, default='data/mfcc_redata_1.5s', help='path of load data')

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
    parser.add_argument('--num_class', type=int, default=2, help='number of class')
    parser.add_argument('--fold', type=int, default=5, help='fold')
    parser.add_argument('--gpu', default="1")
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

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='TB_log/phy2016_1')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                      cooldown=0,
                                                      patience=10,
                                                      min_lr=0.5e-6)
    my_callbacks = [lr_reducer, tensorboard]
    acc_list = []
    spe_list = []
    se_list = []
    macc_list = []
    pr_list = []
    f1_list = []

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
        #acc = accuracy_score(test_label, test_predictions)
        #spe = specificity(test_label, test_predictions)
        #rec = recall_score(test_label, test_predictions, average='macro')
        #macc = (spe+rec)/2.0
        #pre = precision_score(test_label, test_predictions, average='macro')
        #f1 = f1_score(test_label, test_predictions, average='macro')
        acc, spe, se, macc, pr, f1 = acc_sp_se_macc_pr_f1(test_label, test_predictions)
        acc_list.append(acc)
        spe_list.append(spe)
        se_list.append(se)
        macc_list.append(macc)
        pr_list.append(pr)
        f1_list.append(f1)
        file = open(args.save_path, "a")
        file.write(str(j) + ': ' + str(acc) + ' ' + str(spe) + ' ' + str(se) + ' ' + str(macc) +
                   str(pr) + ' ' + str(f1) + '\n')
        file.close()
        print("result save success!")
        model.save(args.save_model.format(j))
        print("save success")
    print("model train success!")
    file = open(args.save_path, "a")
    file.write(
        '1.5s phy2016 5fold:' + '\n'
        'mean:' + str(np.mean(acc_list)) + ' ' + str(np.mean(spe_list)) + ' ' + str(np.mean(se_list)) + ' ' + str(
            np.mean(macc_list)) + str(np.mean(pr_list)) + ' ' + str(np.mean(f1_list)) + '\n')
    file.close()
    print("result save success!")
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
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import math
import os
from model.model_1dcnn import build_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def specificity(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    tp = con_mat[1, 1]
    fn = con_mat[1, 0]
    fp = con_mat[0, 1]
    tn = con_mat[0, 0]
    spe = tn / (tn + fp)
    return spe

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
    parser.add_argument('--save_path', type=str, default='result/txt/only1dcrnn.txt', help='path of saved result')
    parser.add_argument('--save_model', type=str, default='result/model/only1dcrnn_{}.h5', help='path of saved model result')
    parser.add_argument('--train_data1_path', type=str, default='train_data1.npy', help='path of load data')
    parser.add_argument('--train_data2_path', type=str, default='train_data2.npy', help='path of load data')
    parser.add_argument('--train_label_path', type=str, default='train_label.npy', help='path of load label')
    parser.add_argument('--val_data1_path', type=str, default='val_data1.npy', help='path of load data')
    parser.add_argument('--val_data2_path', type=str, default='val_data2.npy', help='path of load data')
    parser.add_argument('--val_label_path', type=str, default='val_label.npy', help='path of load label')
    parser.add_argument('--test_data1_path', type=str, default='test_data1.npy', help='path of load data')
    parser.add_argument('--test_data2_path', type=str, default='test_data2.npy', help='path of load data')
    parser.add_argument('--test_label_path', type=str, default='test_label.npy', help='path of load label')
    parser.add_argument('--data_path', type=str, default='data/mfcc_redata_1.5s', help='path of load data')
    parser.add_argument('--high', type=int, default=149, help='high of 2d feature')
    parser.add_argument('--wide', type=int, default=39, help='high of 2d feature')
    parser.add_argument('--num_epochs', type=int, default=100, help='epuch')
    parser.add_argument('--num_class', type=int, default=2, help='number of class')
    parser.add_argument('--fold', type=int, default=1, help='fold')
    parser.add_argument('--gpu', default="1")
    parser.add_argument('--length', type=int, default=3000)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print('begin load train_data1...')
    train_data1 = np.load(os.path.join(args.data_path, args.train_data1_path), mmap_mode='r')
    print('load train label!')
    train_label = np.load(os.path.join(args.data_path, args.train_label_path), mmap_mode='r')

    print('load val data1!')
    val_data1 = np.load(os.path.join(args.data_path, args.val_data1_path), mmap_mode='r')
    print('load val label')
    val_label = np.load(os.path.join(args.data_path, args.val_label_path), mmap_mode='r')

    print('load test data1')
    test_data1 = np.load(os.path.join(args.data_path, args.test_data1_path))
    print('load test label')
    test_label = np.load(os.path.join(args.data_path, args.test_label_path))
    print("begin train...")

    #tensorboard = tf.keras.callbacks.TensorBoard(log_dir='TB_log/mfcc5')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                      cooldown=0,
                                                      patience=10,
                                                      min_lr=0.5e-6)
    my_callbacks = lr_reducer
    acc_list = []
    spe_list = []
    rec_list = []
    macc_list = []
    pre_list = []
    f1_list = []
    #batch_size = 64
    #batch_num =args.num_train//batch_size

    #train_data_generator = MakeDataset(train_data, train_label, batch_size)
    #var_data_generator = MakeDataset(val_data, val_label, batch_size)

    for j in range(args.fold):
        print("this is fold:", j)
        model = build_model(args.num_class, args.length)
        print('begin train')
        #mini_train = minibatch(train_data, train_label, batch_size)
        #mini_val = minibatch(val_data, val_label, batch_size)
        #weigh={0:.3,1:.7}
        model.fit(train_data1, train_label,
                  epochs=args.num_epochs,
                  batch_size=32,
                  #class_weight=weigh,
                  validation_data=(val_data1, val_label),
                  verbose=1,
                  callbacks=my_callbacks
                  )
        test_predictions = model.predict(test_data1)
        test_predictions = predict(test_predictions)
        acc = accuracy_score(test_label, test_predictions)
        spe = specificity(test_label, test_predictions)
        rec = recall_score(test_label, test_predictions, average='macro')
        macc = (spe+rec)/2.0
        pre = precision_score(test_label, test_predictions, average='macro')
        f1 = f1_score(test_label, test_predictions, average='macro')
        acc_list.append(acc)
        spe_list.append(spe)
        rec_list.append(rec)
        macc_list.append(macc)
        pre_list.append(pre)
        f1_list.append(f1)
        file = open(args.save_path, "a")
        file.write(str(j) + ': ' + str(acc) + ' ' + str(spe) + ' ' + str(rec) + ' ' + str(macc) +
                   str(pre) + ' ' + str(f1) + '\n')
        file.close()
        print("result save success!")
        model.save(args.save_model.format(j))
        print("save success")
    print("model train success!")
    file = open(args.save_path, "a")
    file.write(
        '1.5s redata mfcc only 1dcnn:' + '\n'
        'mean:' + str(np.mean(acc_list)) + ' ' + str(np.mean(spe_list)) + ' ' + str(np.mean(rec_list)) + ' ' + str(
            np.mean(macc_list)) + str(np.mean(pre_list)) + ' ' + str(np.mean(f1_list)) + '\n')
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
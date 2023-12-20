from __future__ import division
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from utilities import preprocess_images,preprocess_label
import time
from model import PSCTmodel
import h5py, yaml
import math
from argparse import ArgumentParser
import os

from scipy import stats

def evaluationmetrics(_y,_y_pred):
    sq = np.reshape(np.asarray(_y), (-1,))
    q = np.reshape(np.asarray(_y_pred), (-1,))
    plcc = stats.pearsonr(sq, q)[0] # you can also calculate it using MATLAB
    srcc = stats.spearmanr(q, sq)[0] # you can also calculate it using MATLAB
    return plcc,srcc
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def datasetgenerator(conf, log_dir, EXP_ID ='0'):

    datainfo = conf['datainfo']
    Info = h5py.File(datainfo)
    index = Info['index1']
    ref_ids = Info['ref_ids'][0, :]
    testindex = index[:int(math.ceil(1 * len(index)))]
    trainindex = index[int(math.ceil((1 - 1) * len(index))):]
    test_index,train_index, val_index = [], [], []
    for i in range(len(ref_ids)):
        test_index.append(i) if (ref_ids[i] in testindex) else \
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                val_index.append(i)
    print('test_index:', test_index)

    print(len(test_index))
    if len(test_index) > 0:
        ensure_dir(log_dir)
        testTfile = log_dir + EXP_ID + '.txt'
        outfile = open(testTfile, "w")
        print("\n".join(str(Info['SS_all'][0, i]) for i in test_index), file=outfile)
        outfile.close()
    return test_index

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs,conf, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.conf = conf
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_IDs_temp.sort()
        # Generate data
        XS,XH,Xsal,Y  = self.__data_generation(list_IDs_temp) # XS:disimg XH:refimg XSal:saliency images Y:MOS
        return [XS,XH,Xsal], [Y]


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        srim_dir = self.conf['srim_dir']
        hrim_dir = self.conf['hrim_dir']
        salim_dir = self.conf['salim_dir']
        datainfo = self.conf['datainfo']
        shape_r =  self.conf['shape_r']
        shape_c =  self.conf['shape_c']
        Info = h5py.File(datainfo)
        index = list_IDs_temp

        mos = Info['SS_all'][0, index]

        srim_names = [Info[Info['filenamesdis'][0, :][i]].value.tobytes()\
                                [::2].decode() for i in index]
        hrim_names = [Info[Info['filenamesref'][0, :][i]].value.tobytes() \
                          [::2].decode() for i in index]
        salim_names = [Info[Info['filenamesdis'][0, :][i]].value.tobytes() \
                          [::2].decode() for i in index] #The name of the saliency map is the same as that of the distorted image.

        srimages = [os.path.join(srim_dir, srim_names[idx]) for idx in range(len(index))]
        hrimages = [os.path.join(hrim_dir, hrim_names[idx]) for idx in range(len(index))]
        salimages = [os.path.join(salim_dir, salim_names[idx]) for idx in range(len(index))]
        maps = [mos[idx] for idx in range(len(index))]
        XS,XH,Xsal = preprocess_images(srimages[0:len(index)], hrimages[0:len(index)], salimages[0:len(index)],shape_r, shape_c)
        Y = preprocess_label(maps[0:len(index)])

        return XS,XH,Xsal,Y


if __name__ == '__main__':

    parser = ArgumentParser() # You can change parameters as needed.
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size for training')
    parser.add_argument('--maxepochs', type=int, default=800,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--config', default='config_onlyfortest.yaml', type=str,
                        help='config file path')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id')
    parser.add_argument('--database', default='exampleonlyfortest', type=str,
                        help='database name')
    parser.add_argument('--phase', default='test', type=str,
                        help='test')
    args = parser.parse_args()

    with open(args.config,'rb') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    print('phase: ' + args.phase)
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('lr: ' + str(args.lr))
    print('batch_size: ' + str(args.batch_size))
    config.update(config[args.database])

    log_dir = '.\data' + '/EXP{}-{}-testtxt/'.format(args.exp_id, args.database)

    b_s= args.batch_size
    img_rows = config['shape_r']
    img_cols = config['shape_c']
    model = PSCTmodel(img_rows=img_rows,img_cols=img_cols)
    print("Compile PSCT Model")

    opt = Adam(lr=args.lr)
    model.compile(optimizer=opt, loss= ['mae'])
    model.summary()
    test_index = datasetgenerator(config, log_dir, args.exp_id)


    if args.phase == "test":
        arg = 'example.h5'
        print("Load weights")
        # weight_file = '../checkpoint/'+ arg
        weight_file = './checkpoint_example/' + arg
        print('weight_file:', weight_file)
        model.load_weights(weight_file)
        print('load done')
        output_folder = 'TestResults_example/' + arg + args.database + '/'
        if os.path.isdir(output_folder) is False:
            os.makedirs(output_folder)

        nb_imgs_test = len(test_index)
        print(nb_imgs_test)
        totalrsults = [0] * nb_imgs_test
        output_folderfileavg = output_folder + 'results_avg' + '.txt'
        start_time0 = time.time()
        repeat = 1
        for i in range(repeat):
            output_folderfile = output_folder + 'results' + str(i) + '.txt'

            start_time = time.time()
            test_generator = DataGenerator(test_index, config, 1, shuffle=False)
            predictions = model.predict_generator(test_generator, nb_imgs_test)
            predictions0 = predictions

            results = []
            for pred in predictions0:
                results.append(float(pred))

            outfile = open(output_folderfile, "w")
            print("\n".join(str(i) for i in results), file=outfile)
            outfile.close()
            totalrsults = [sum(x) for x in zip(results, totalrsults)]
        totalrsults = [x / repeat for x in totalrsults]
        outfile = open(output_folderfileavg, "w")
        print("\n".join(str(i) for i in totalrsults), file=outfile)
        outfile.close()


        with open(output_folderfileavg) as f:
            content = f.readlines()
        maps2 = [float(x.strip()) for x in content]
        f.close()
        testTfile = log_dir + args.exp_id + '.txt'
        print("Predict quality for " + testTfile + " at " + output_folder)
        with open(testTfile) as f:
            content = f.readlines()
        maps = [float(x.strip()) for x in content]
        f.close()

        plcc, srcc = evaluationmetrics(maps, maps2)
        print("Testing Results  :PLCC: {:.4f} SRCC: {:.4f} " .format(plcc, srcc))

    else:
        raise NotImplementedError
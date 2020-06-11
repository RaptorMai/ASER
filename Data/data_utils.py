import os
import sys
from utils import names_match
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
import tarfile
import zipfile
import utils.global_vars as global_vars
from utils.utils import generate_mnist_tasks
from Data.data import *
import time, tqdm, cv2, glob

############################################################
### CIFAR download utils ###################################
############################################################
CIFAR_10_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_100_URL = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR_10_DIR = "/cifar_10"
CIFAR_100_DIR = "/cifar_100"
VAL_SPLIT = 0.1
MNIST_NUM_TASK = -1
DATA_DIR = 'Data/CIFAR_data'

## CUB related options
CUB_DATA_DIR = './Data/CUB_data/CUB_200_2011/images'
CUB_TRAIN_LIST = './Data/CUB_data/dataset_lists/CUB_train_list.txt'
CUB_TEST_LIST = './Data/CUB_data/dataset_lists/CUB_test_list.txt'
CUB_VAL_SPLIT = 2
IMG_MEAN = np.array((103.94,116.78,123.68), dtype=np.float32)

## AWA related options
AWA_DATA_DIR= './Data/AWA_data/Animals_with_Attributes2/'
AWA_TRAIN_LIST = './Data/AWA_data/dataset_lists/AWA_train_list.txt'
AWA_TEST_LIST = './Data/AWA_data/dataset_lists/AWA_test_list.txt'

def generate_data(args, random_seed):
    np.random.seed(random_seed)
    task_labels = []
    if args.data == 'mnist':
        print('Embedded size for MLP: {}'.format(args.embed_dim))
        datareader = readMNIST(datapath="Data/mnist.pkl.gz")
        dataset = datareader.get_dataset()
        params = [("Original", None)] + [('Permuted', i) for i in range(args.num_tasks - 1)]
        result, objects = generate_mnist_tasks(dataset, params)
        trainset, valset, testsets = result
        return trainset, valset, testsets

    elif args.data == 'split_mnist':
        print('Embedded size for MLP: {}'.format(args.embed_dim))
        datareader = readMNIST(datapath="Data/mnist.pkl.gz")
        dataset = datareader.get_dataset()
        label_array = np.arange(0, 10)
        classes_per_task = 10 // args.num_tasks
        for tt in range(args.num_tasks):
            tt_offset = tt * classes_per_task
            task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
            print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
        trainset, valset, testset = construct_split_mnist(dataset, task_labels, random_seed, combine_val=args.combine_val)

    elif args.data == 'cifar100':
        cifar_classes = 100
        classes_per_task = cifar_classes // args.num_tasks
        total_classes = classes_per_task * args.num_tasks
        label_array = np.arange(0, total_classes)
        if not args.fixed_order:
            np.random.shuffle(label_array)
        if args.all_data:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[0:tt_offset + classes_per_task]))
                print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
            trainset, valset, testset = construct_split_cifar(task_labels, random_seed, cifar100=True, combine_val=args.combine_val)
        else:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
                print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
            trainset, valset, testset = construct_split_cifar(task_labels, random_seed, cifar100=True, combine_val=args.combine_val)

    elif args.data == 'miniimagenet':
        num_classes = 100
        classes_per_task = num_classes // args.num_tasks
        total_classes = classes_per_task * args.num_tasks
        label_array = np.arange(0, total_classes)
        if not args.fixed_order:
            np.random.shuffle(label_array)
        if args.all_data:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[0:tt_offset + classes_per_task]))
                print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
            task_labels = np.array(task_labels)
            trainset, valset, testset = split_miniimagenet(task_labels, random_seed, combine_val=args.combine_val)
        else:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
                print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
            trainset, valset, testset = split_miniimagenet(task_labels, random_seed, combine_val=args.combine_val)

    elif args.data == 'cifar10':
        cifar_classes = 10
        classes_per_task = cifar_classes // args.num_tasks
        total_classes = classes_per_task * args.num_tasks
        label_array = np.arange(0, total_classes)
        if not args.fixed_order:
            np.random.shuffle(label_array)
        if args.all_data:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[0:tt_offset + classes_per_task]))
                print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
            trainset, valset, testset = construct_split_cifar(task_labels, random_seed, cifar100=False, combine_val=args.combine_val)
        else:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
                print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
            trainset, valset, testset = construct_split_cifar(task_labels, random_seed, cifar100=False, combine_val=args.combine_val)

    elif args.data.lower() == 'cub':
        cub_classes = 200
        classes_per_task = cub_classes // args.num_tasks
        total_classes = classes_per_task * args.num_tasks
        label_array = np.arange(0, total_classes)
        if not args.fixed_order:
            np.random.shuffle(label_array)
        if args.all_data:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[0: tt_offset + classes_per_task]))
                print('Task: {}, Labels: {}'.format(tt, task_labels[tt]))
            trainset, valset, testset = construct_split_cub(task_labels, random_seed, combine_val=args.combine_val)
        else:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
                print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
            trainset, valset, testset = construct_split_cub(task_labels, random_seed, combine_val=args.combine_val)

    elif args.data.lower() == 'awa':
        awa_classes = 50
        classes_per_task = awa_classes // args.num_tasks    # 10 classes per task in Chaudhry et al. (2019a, 2019b)
        total_classes = classes_per_task * args.num_tasks
        label_array = np.arange(0, total_classes)
        if not args.fixed_order:
            np.random.shuffle(label_array)
        if args.all_data:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[0: tt_offset + classes_per_task]))
                print('Task: {}, Labels: {}'.format(tt, task_labels[tt]))
            trainset, valset, testset = construct_split_awa(task_labels)
        else:
            for tt in range(args.num_tasks):
                tt_offset = tt * classes_per_task
                task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))
                print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
            trainset, valset, testset = construct_split_awa(task_labels)

    elif args.data == 'steel':
        data_path = 'Data/steel'
        data = load_steel(data_path)
        task_labels = [[0, 1, 2], [0, 3, 4]]
        trainset, testset = construct_split_steel(data, task_labels)

    elif 'NI' in args.data:
        print('Variance: {}'.format(args.variance))
        print('factor: {}'.format(args.change_factor))
        if args.variance in global_vars.IMBALANCE:
            trainset, testset = construct_Imbalance_NI(args.variance, args.change_factor, random_seed,
                                                         num_tasks=args.num_tasks,
                                                        data=args.data)
        else:
            trainset, testset = construct_Nstationary_NI(random_seed, args.variance, args.change_factor,
                                                         num_tasks=args.num_tasks,
                                                         task_list=args.task_composition, data=args.data)
        if args.all_data:
            all_train = []
            for idx, val in enumerate(trainset):

                if idx == 0:
                    all_train.append(val)
                else:
                    x = val.images
                    y = val.labels
                    x_prev = all_train[idx - 1].images
                    y_prev = all_train[idx - 1].labels
                    new_x = np.concatenate((x, x_prev), axis=0)
                    new_y = np.concatenate((y, y_prev), axis=0)
                    perm_inds = np.arange(0, new_x.shape[0])
                    np.random.shuffle(perm_inds)
                    new_x = new_x[perm_inds]
                    new_y = new_y[perm_inds]
                    print(new_x.shape, new_y.shape)
                    all_train.append(imageSet(new_x, new_y))
            trainset = all_train
    else:
        raise Exception('wrong dataset')
    return trainset, valset, testset, task_labels

def _shuffle(x, y, random_seed):
    rng = np.random.RandomState(random_seed)
    perm_inds = np.arange(0, x.shape[0])
    rng.shuffle(perm_inds)
    rdm_x = x[perm_inds]
    rdm_y = y[perm_inds]
    return rdm_x, rdm_y

def split_miniimagenet(task_labels, random_seed, combine_val=False):
    train_in = open("Data/miniimagenet/mini-imagenet-cache-train.pkl", "rb")
    train = pickle.load(train_in)
    train_x = train["image_data"].reshape([64, 600, 84, 84, 3])
    val_in = open("Data/miniimagenet/mini-imagenet-cache-val.pkl", "rb")
    val = pickle.load(val_in)
    val_x = val['image_data'].reshape([16, 600, 84, 84, 3])
    test_in = open("Data/miniimagenet/mini-imagenet-cache-test.pkl", "rb")
    test = pickle.load(test_in)
    test_x = test['image_data'].reshape([20, 600, 84, 84, 3])
    all_data = np.vstack((train_x, val_x, test_x))
    train_list = []
    test_list = []
    val_list = []
    for task in task_labels:
        cur_data = all_data[task]
        cur_all = cur_data.reshape((-1, 84, 84, 3))
        cur_y = np.repeat(task, 600)
        x_random, y_random = _shuffle(cur_all, cur_y, 0)
        l = x_random.shape[0]
        if combine_val:
            x_test = x_random[: int(l * VAL_SPLIT)]
            y_test = y_random[: int(l * VAL_SPLIT)]
            x_train = x_random[int(l * VAL_SPLIT):]
            y_train = y_random[int(l * VAL_SPLIT):]
            y_test_oh = np.eye(100)[y_test]
            y_train_oh = np.eye(100)[y_train]
            test_list.append(imageSet(x_test, y_test_oh))
            train_list.append(imageSet(x_train, y_train_oh))
        else:
            x_val = x_random[: int(l * VAL_SPLIT)]
            y_val = y_random[: int(l * VAL_SPLIT)]
            x_test = x_random[int(l * VAL_SPLIT): int(l * 2*VAL_SPLIT)]
            y_test = y_random[int(l * VAL_SPLIT): int(l * 2*VAL_SPLIT)]
            x_train = x_random[int(l * 2*VAL_SPLIT):]
            y_train = y_random[int(l * 2*VAL_SPLIT):]
            y_test_oh = np.eye(100)[y_test]
            y_train_oh = np.eye(100)[y_train]
            y_val_oh = np.eye(100)[y_val]
            val_list.append(imageSet(x_val, y_val_oh))
            test_list.append(imageSet(x_test, y_test_oh))
            train_list.append(imageSet(x_train, y_train_oh))
    return train_list, val_list, test_list

def get_NI_data(random_seed, data='cifar10-NI'):
    if data == 'cifar10-NI':
        cifar_data = _get_cifar(DATA_DIR, False, True)
        train_x = cifar_data['train'][0]
        train_y = cifar_data['train'][1]
        test_x = cifar_data['test'][0]
        test_y = cifar_data['test'][1]
    elif data == 'steel-NI':
        data = load_steel('Data/steel')
        train_x = data['train']['images']
        train_y = data['train']['labels']
        test_x = data['test']['images']
        test_y = data['test']['labels']
    np.random.seed(random_seed)
    perm_inds = np.arange(0, train_x.shape[0])
    perm_inds_test = np.arange(0, test_x.shape[0])
    np.random.shuffle(perm_inds)
    np.random.shuffle(perm_inds_test)
    rdm_train_x = train_x[perm_inds]
    rdm_train_y = train_y[perm_inds]
    rdm_test_x = test_x[perm_inds_test]
    rdm_test_y = test_y[perm_inds_test]
    return rdm_train_x, rdm_train_y, rdm_test_x, rdm_test_y


def construct_Imbalance_NI(variance, change_factor, random_seed=0, num_tasks=10, data='cifar10-NI'):
    rdm_train_x, rdm_train_y, rdm_test_x, rdm_test_y = get_NI_data(random_seed, data)
    train_x_split = rdm_train_x.reshape(num_tasks, -1, 32, 32, 3)
    train_y_split = rdm_train_y.reshape(num_tasks, -1, 10)
    test_x_split = rdm_test_x.reshape(num_tasks, -1, 32, 32, 3)
    test_y_split = rdm_test_y.reshape(num_tasks, -1, 10)
    # Data splits
    train_list = []
    test_list = []
    # fig, ax_list = plt.subplots(num_tasks, figsize=(20,60))
    change = names_match.non_stationary[variance]
    for i in range(num_tasks):
        tmp = change(train_x_split[i], train_y_split[i])
        train_list.append(tmp.next_task(random_seed + i, change_factor))
        # tmp.show_distribution('train', ax_list[i], 'Task_{}'.format(str(i)))

        tmp_test = change(test_x_split[i], test_y_split[i])
        test_list.append(tmp_test.next_task(random_seed + i, change_factor))
        # tmp_test.show_distribution('test', ax_list[i], 'Task_{}'.format(str(i)))
    return train_list, test_list

def construct_Nstationary_NI(random_seed, variance, change_factor, task_list=(2,3,2,3), num_tasks=10, data='cifar10-NI', plot=False):
    if data == 'cifar10-NI':
        rdm_train_x, rdm_train_y, rdm_test_x, rdm_test_y = get_NI_data(random_seed, data)
        train_x_split = rdm_train_x.reshape(num_tasks, -1, 32, 32, 3)
        train_y_split = rdm_train_y.reshape(num_tasks, -1, 10)
        test_x_split = rdm_test_x.reshape(num_tasks, -1, 32, 32, 3)
        test_y_split = rdm_test_y.reshape(num_tasks, -1, 10)
    elif data == 'steel-NI':
        rdm_train_x, rdm_train_y, rdm_test_x, rdm_test_y = get_NI_data(random_seed, data)
        offset_train = rdm_train_x.shape[0] % num_tasks
        offset_test = rdm_test_x.shape[0] % num_tasks
        train_x_split = rdm_train_x[offset_train:].reshape(num_tasks, -1, 128, 128, 1)
        train_y_split = rdm_train_y[offset_train:].reshape(num_tasks, -1, 5)
        test_x_split = rdm_test_x[offset_test:].reshape(num_tasks, -1, 128, 128, 1)
        test_y_split = rdm_test_y[offset_test:].reshape(num_tasks, -1, 5)
    else:
        raise Exception('wrong data')

    # Data splits
    train_list = []
    test_list = []
    change = names_match.non_stationary[variance]
    orig = Data
    i = 0
    if len(change_factor) == 1:
        change_factor = change_factor[0]
    for idx, val in enumerate(task_list):
        if idx % 2 == 0:
            for _ in range(val):
                #train
                if data in global_vars.COLOR:
                    tmp = orig(train_x_split[i], train_y_split[i], color=True)
                else:
                    tmp = orig(train_x_split[i], train_y_split[i])
                train_list.append(tmp.next_task())

                #test
                if plot:
                    tmp.show_sample()
                if data in global_vars.COLOR:
                    tmp_test = orig(test_x_split[i], test_y_split[i], color=True)
                else:
                    tmp_test = orig(test_x_split[i], test_y_split[i])
                test_list.append(tmp_test.next_task())
                print(i, 'normal')
                i += 1
        else:
            for _ in range(val):
                #train
                if data in global_vars.COLOR:
                    tmp = change(train_x_split[i], train_y_split[i], color=True)
                else:
                    tmp = change(train_x_split[i], train_y_split[i])
                train_list.append(tmp.next_task(change_factor))
                if plot:
                    tmp.show_sample()
                #test
                if data in global_vars.COLOR:
                    tmp_test = change(test_x_split[i], test_y_split[i], color=True)
                else:
                    tmp_test = change(test_x_split[i], test_y_split[i])
                test_list.append(tmp_test.next_task(change_factor))
                if plot:
                    tmp_test.show_sample()
                print(i, 'change')
                i += 1
    return train_list, test_list


def construct_split_cifar(task_labels, random_seed, cifar100=True, combine_val=True):
    """
    Construct Split CIFAR-10 and CIFAR-100 datasets
    Args:
        task_labels     Labels of different tasks
        data_dir        Data directory where the CIFAR data will be saved
    """


    # Get the cifar dataset
    cifar_data = _get_cifar(DATA_DIR, is_cifar_100=cifar100)


    # Data splits
    train_list = []
    test_list = []
    val_list = []

    for task in task_labels:
        data = cifar_data['train']
        x_train, y_train = load_task_task_labels(data, task)
        if combine_val:
            train_list.append(imageSet(x_train, y_train))
        else:
            x_random, y_random = _shuffle(x_train, y_train, random_seed)
            l = x_random.shape[0]
            x_val = x_random[: int(l*VAL_SPLIT)]
            y_val = y_random[: int(l*VAL_SPLIT)]
            x_train = x_random[int(l*VAL_SPLIT):]
            y_train = y_random[int(l*VAL_SPLIT):]
            val_list.append(imageSet(x_val, y_val))
            train_list.append(imageSet(x_train, y_train))

        data = cifar_data['test']
        x_test, y_test = load_task_task_labels(data, task)
        test_list.append(imageSet(x_test, y_test))

    return train_list, val_list, test_list

def _get_cifar(data_dir, is_cifar_100, raw=False):
    """
    Get the CIFAR-10 and CIFAR-100 datasets
    Args:
        data_dir        Directory where the downloaded data will be stored
    """
    x_train = None
    y_train = None
    x_validation = None
    y_validation = None
    x_test = None
    y_test = None
    l = None

    # Download the dataset if needed
    _cifar_maybe_download_and_extract(data_dir)

    # Dictionary to store the dataset
    dataset = dict()
    dataset['train'] = []
    dataset['test'] = []

    def dense_to_one_hot(labels_dense, num_classes=100):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot

    if is_cifar_100:
        # Load the training data of CIFAR-100
        f = open(data_dir + CIFAR_100_DIR + '/train', 'rb')
        datadict = pickle.load(f, encoding='iso-8859-1')
        f.close()

        _X = datadict['data']
        _Y = np.array(datadict['fine_labels'])
        _Y = dense_to_one_hot(_Y, num_classes=100)

        _X = _X.reshape([-1, 3, 32, 32])
        _X = _X.transpose([0, 2, 3, 1])

        if not raw:
            _X = np.array(_X, dtype=float) / 255.0
        # Compute the data mean for normalization
        x_train_mean = np.mean(_X, axis=0)

        x_train = _X
        y_train = _Y


    else:
        # Load all the training batches of the CIFAR-10
        for i in range(5):
            f = open(data_dir + CIFAR_10_DIR + '/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f,encoding='iso-8859-1')
            f.close()

            _X = datadict['data']
            _Y = np.array(datadict['labels'])
            _Y = dense_to_one_hot(_Y, num_classes=10)

            if not raw:
                _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            _X = _X.transpose([0, 2, 3, 1])

            if x_train is None:
                x_train = _X
                y_train = _Y
            else:
                x_train = np.concatenate((x_train, _X), axis=0)
                y_train = np.concatenate((y_train, _Y), axis=0)

        # Compute the data mean for normalization
        x_train_mean = np.mean(x_train, axis=0)

    if not raw:
        # Normalize the train and validation sets
        x_train -= x_train_mean

    dataset['train'].append(x_train)
    dataset['train'].append(y_train)
    dataset['train'].append(l)


    if is_cifar_100:
        # Load the test batch of CIFAR-100
        f = open(data_dir + CIFAR_100_DIR + '/test', 'rb')
        datadict = pickle.load(f, encoding='iso-8859-1')
        f.close()

        _X = datadict['data']
        _Y = np.array(datadict['fine_labels'])
        _Y = dense_to_one_hot(_Y, num_classes=100)
    else:
        # Load the test batch of CIFAR-10
        f = open(data_dir + CIFAR_10_DIR + '/test_batch', 'rb')
        datadict = pickle.load(f, encoding='iso-8859-1')
        f.close()

        _X = datadict["data"]
        _Y = np.array(datadict['labels'])
        _Y = dense_to_one_hot(_Y, num_classes=10)

    if not raw:
        _X = np.array(_X, dtype=float) / 255.0
    _X = _X.reshape([-1, 3, 32, 32])
    _X = _X.transpose([0, 2, 3, 1])

    x_test = _X
    y_test = _Y

    if not raw:
        # Normalize the test set
        x_test -= x_train_mean

    dataset['test'].append(x_test)
    dataset['test'].append(y_test)
    dataset['test'].append(l)

    return dataset


def _print_download_progress(count, block_size, total_size):
    """
    Show the download progress of the cifar data
    """
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def _cifar_maybe_download_and_extract(data_dir):
    """
    Routine to download and extract the cifar dataset
    Args:
        data_dir      Directory where the downloaded data will be stored
    """
    cifar_10_directory = data_dir + CIFAR_10_DIR
    cifar_100_directory = data_dir + CIFAR_100_DIR

    # If the data_dir does not exist, create the directory and download
    # the data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

        url = CIFAR_10_URL
        filename = url.split('/')[-1]
        file_path = os.path.join(data_dir, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)
        print("Done.")

        url = CIFAR_100_URL
        filename = url.split('/')[-1]
        file_path = os.path.join(data_dir, filename)
        zip_cifar_100 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(data_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(data_dir)
        print("Done.")

        os.rename(data_dir + "/cifar-10-batches-py", cifar_10_directory)
        os.rename(data_dir + "/cifar-100-python", cifar_100_directory)
        os.remove(zip_cifar_10)
        os.remove(zip_cifar_100)


def load_task_specific_data(datasets, task_labels):
    """
    Loads task specific data from the datasets
    """
    global_class_indices = np.column_stack(np.nonzero(datasets['labels']))
    count = 0
    for cls in task_labels:
        if count == 0:
            class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] == cls][:,np.array([True, False])])
        else:
            class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:,1] == cls][:,np.array([True, False])]))
        count += 1
    class_indices = np.sort(class_indices, axis=None)

    return imageSet(datasets['images'][class_indices, :], datasets['labels'][class_indices, :])

def load_task_task_labels(dataset, task_labels):
    global_class_indices = np.column_stack(np.nonzero(dataset[1]))
    count = 0

    for cls in task_labels:
        if count == 0:
            class_indices = np.squeeze(global_class_indices[global_class_indices[:, 1] ==
                                                            cls][:, np.array([True, False])])
        else:
            class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:, 1] == \
                                                                                     cls][:, np.array([True, False])]))

        count += 1

    class_indices = np.sort(class_indices, axis=None)

    # datasets[idx].append(imageSet(deepcopy(this_set[0][class_indices, :]), deepcopy(this_set[1][class_indices, :])))
    return deepcopy(dataset[0][class_indices, :]), deepcopy(dataset[1][class_indices, :]) #imageSet(deepcopy(dataset[0][class_indices, :]), deepcopy(dataset[1][class_indices, :]))


def construct_split_steel(dataset, task_labels):
    # Define a list for storing the data for different tasks
    datasets = [[] for _ in range(2)]

    # Data splits
    sets = ["train", "test"]
    for task in task_labels:

        for idx, set_name in enumerate(sets):
            this_set = dataset[set_name]

            global_class_indices = np.column_stack(np.nonzero(this_set['labels']))
            count = 0

            for cls in task:
                if count == 0:
                    class_indices = np.squeeze(global_class_indices[global_class_indices[:,1] ==
                                                                    cls][:,np.array([True, False])])
                else:
                    class_indices = np.append(class_indices, np.squeeze(global_class_indices[global_class_indices[:,1] == \
                                                                                             cls][:,np.array([True, False])]))

                count += 1

            class_indices = np.sort(class_indices, axis=None)


            datasets[idx].append(imageSet(deepcopy(this_set['images'][class_indices, :]), deepcopy(this_set['labels'][class_indices, :])))
    return datasets[0], datasets[1]

def construct_split_mnist(dataset, task_labels, random_seed, combine_val=True):
    np.random.seed(random_seed)
    dataset = [list(i) for i in dataset]
    if combine_val:
        dataset[0][0] = np.vstack((dataset[0][0], dataset[1][0]))
        dataset[0][1] = np.append(dataset[0][1], dataset[1][1])
    ret = [[] for _ in range(3)]
    sets = ["train", "val", "test"]

    for task in task_labels:

        for idx, set_name in enumerate(sets):
            this_set = dataset[idx]
            if combine_val and set_name == 'val':
                continue
            count = 0
            for cls in task:
                if count == 0:
                    indices = np.nonzero(this_set[1] == cls)[0]
                else:
                    indices = np.append(indices, np.nonzero(this_set[1] == cls)[0])

                count += 1
            indices = np.random.permutation(indices)[:MNIST_NUM_TASK]
            ret[idx].append(imageSet(deepcopy(this_set[0][indices, :]), np.eye(10)[this_set[1][indices]]))
    return ret[0], ret[1], ret[2]

def construct_split_awa(task_labels):
    """
    Construct Split AWA dataset
    :param task_labels:         (list) List of class labels per task
    :return:
    """
    # height and width of images are fixed to 224 x 224 as in ImageNet
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    # Data splits
    train_list = []
    test_list = []
    val_list = []

    # get the AWA dataset (simply load the resized images after first run)
    awa_data = _get_AWA(AWA_DATA_DIR, AWA_TRAIN_LIST, AWA_TEST_LIST, IMG_HEIGHT, IMG_WIDTH)

    # define a list for storing the data for different tasks
    print("\nSet up split tasks..")
    st = time.time()
    for task in tqdm.tqdm(task_labels):
        data = awa_data['train']
        x_train, y_train = load_task_task_labels(data, task)
        train_list.append(imageSet(x_train, y_train))

        data = awa_data['test']
        x_test, y_test = load_task_task_labels(data, task)
        test_list.append(imageSet(x_test, y_test))
    et = time.time()
    print("Done. Time taken: {:.4f}s".format(et - st))
    return train_list, val_list, test_list

def construct_split_cub(task_labels, random_seed, combine_val=True):
    """
    Construct Split CUB dataset
    :param task_labels:         (list) List of class labels per task
    :param random_seed:         (int) Random seed
    :param combine_val:         (bool) Whether to combine validation set to train set
    :return:
    """
    # height and width of images are fixed to 224 x 224 as in ImageNet
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNEL = 3

    # Data splits
    train_list = []
    test_list = []
    val_list = []

    # get the CUB dataset (simply load the resized images after first run)
    # `TODO
    cub_data = _get_CUB(CUB_DATA_DIR, CUB_TRAIN_LIST, CUB_TEST_LIST, IMG_HEIGHT, IMG_WIDTH)

    # define a list for storing the data for different tasks
    print("\nSet up split tasks..")
    st = time.time()
    for task in tqdm.tqdm(task_labels):
        data = cub_data['train']
        x_train, y_train = load_task_task_labels(data, task)
        num_class_task = len(task)
        if combine_val:
            train_list.append(imageSet(x_train, y_train))
        else:
            x_random, y_random = _shuffle(x_train, y_train, random_seed)
            l = x_random.shape[0]
            x_val = x_random[:num_class_task*CUB_VAL_SPLIT]
            y_val = y_random[:num_class_task*CUB_VAL_SPLIT]
            x_train = x_random[num_class_task*CUB_VAL_SPLIT:]
            y_train = y_random[num_class_task*CUB_VAL_SPLIT:]
            val_list.append(imageSet(x_val, y_val))
            train_list.append(imageSet(x_train, y_train))

        data = cub_data['test']
        x_test, y_test = load_task_task_labels(data, task)
        test_list.append(imageSet(x_test, y_test))
    et = time.time()
    print("Done. Time taken: {:.4f}s".format(et-st))
    return train_list, val_list, test_list

def _get_AWA(data_dir, train_list_file, test_list_file, img_height, img_width):
    saved_path = './Data/AWA_data/dataset'
    num_classes = 50
    st = time.time()
    def convert_to_onehot(dataset):
        for set_name in dataset:
            I = np.identity(num_classes)
            dataset[set_name][1] = I[dataset[set_name][1]]
        return dataset

    if len(glob.glob('./Data/AWA_data/*.npy')) != 0:
        print("Read pre-processed .npy file..")
        for i in range(10):
            assert os.path.isfile(saved_path+str(i)+'.npy'), "File broken: please remove .npy files and re-run"

        dataset = {'train': [], 'test': []}
        for i in range(10):
            dataset_i = np.load(saved_path + str(i) + '.npy', allow_pickle=True).item()
            for set_name in dataset:
                if i == 0:
                    dataset[set_name] = dataset_i[set_name]
                else:
                    for j in range(2):
                        dataset[set_name][j] = np.concatenate((dataset[set_name][j], dataset_i[set_name][j]), axis=0)
        dataset = convert_to_onehot(dataset)
        et = time.time()
        print("Time taken to read file: {:.4f}s".format(et-st))
        return dataset

    dataset = {'train': [], 'test': []}

    # read train, validation and test files
    print("Read train set files and resize..")
    train_img, train_label = _AWA_read_img_from_file(data_dir, train_list_file, img_height, img_width)
    print("Read test set files and resize..")
    test_img, test_label = _AWA_read_img_from_file(data_dir, test_list_file, img_height, img_width)

    dataset['train'].append(train_img)
    dataset['train'].append(train_label)
    dataset['test'].append(test_img)
    dataset['test'].append(test_label)

    print("saving .npy file for later use..")
    data_tr = dataset['train']
    data_te = dataset['test']
    num_tr = data_tr[0].shape[0]
    num_te = data_te[0].shape[0]
    num_per_file_tr = num_tr // 10
    num_per_file_te = num_te // 10
    sid_tr = 0
    sid_te = 0
    for i in range(10):
        eid_tr = sid_tr + num_per_file_tr if i != 9 else num_tr
        eid_te = sid_te + num_per_file_te if i != 9 else num_te
        train_x_i, train_y_i = data_tr[0][sid_tr: eid_tr], data_tr[1][sid_tr: eid_tr]
        test_x_i, test_y_i = data_te[0][sid_te: eid_te], data_te[1][sid_te: eid_te]
        dataset_i = {'train': [train_x_i, train_y_i], 'test': [test_x_i, test_y_i]}
        np.save(saved_path+str(i)+'.npy', dataset_i)
        sid_tr, sid_te = eid_tr, eid_te

    # convert labels to one-hot encoding
    dataset = convert_to_onehot(dataset)
    et = time.time()
    print("Time taken for reading files: {:.4f}s".format(et - st))
    return dataset

def _get_CUB(data_dir, train_list_file, test_list_file, img_height, img_width):
    saved_path = './Data/CUB_data/dataset'
    num_classes = 200
    st = time.time()
    def convert_to_onehot(dataset):
        for set_name in dataset:
            I = np.identity(num_classes)
            dataset[set_name][1] = I[dataset[set_name][1]]
        return dataset

    if len(glob.glob('./Data/CUB_data/*.npy')) != 0:
        print("Read pre-processed .npy file..")
        for i in range(4):
            assert os.path.isfile(saved_path+str(i)+'.npy'), "File broken: please remove .npy files and re-run"

        dataset = {'train': [], 'test': []}
        for i in range(4):
            dataset_i = np.load(saved_path+str(i)+'.npy', allow_pickle=True).item()
            for set_name in dataset:
                if i == 0:
                    dataset[set_name] = dataset_i[set_name]
                else:
                    for j in range(2):
                        dataset[set_name][j] = np.concatenate((dataset[set_name][j], dataset_i[set_name][j]), axis=0)

        dataset = convert_to_onehot(dataset)
        et = time.time()
        print("Time taken to read file: {:.4f}s".format(et-st))
        return dataset

    dataset = dict()
    dataset['train'] = []
    dataset['test'] = []

    # Read train and test files
    print("Read train set files and resize..")
    train_img, train_label = _CUB_read_img_from_file(data_dir, train_list_file, img_height, img_width)
    print("Read test set files and resize..")
    test_img, test_label = _CUB_read_img_from_file(data_dir, test_list_file, img_height, img_width)

    # move some data from test set to train set
    rng = np.random.RandomState(0)
    test_idcs = np.arange(test_label.shape[0])
    to_train = np.empty((0, ), dtype=int)
    num_to_train_per_class = 15                  # move 5 images per class (total 1000 images)
    num_rem_test_per_class = 10                 # leave 10
    cls_label, cnts = np.unique(test_label, return_counts=True)

    for i, cnt in zip(cls_label, cnts):
        cls_idcs = np.where(test_label == i)[0]
        num_to_train = cnt - num_rem_test_per_class
        to_train = np.concatenate((to_train, rng.permutation(cls_idcs)[:num_to_train]), axis=0)

    rem_test = np.setdiff1d(test_idcs, to_train)
    img_to_move, label_to_move = test_img[to_train], test_label[to_train]
    train_img = np.concatenate((train_img, img_to_move), axis=0)
    train_label = np.concatenate((train_label, label_to_move), axis=0)
    test_img, test_label = test_img[rem_test], test_label[rem_test]

    dataset['train'].append(train_img)
    dataset['train'].append(train_label)
    dataset['test'].append(test_img)
    dataset['test'].append(test_label)

    print("saving .npy file for later use..")
    data_tr = dataset['train']
    data_te = dataset['test']
    num_tr = data_tr[0].shape[0]
    num_te = data_te[0].shape[0]
    num_per_file_tr = num_tr // 4
    num_per_file_te = num_te // 4
    sid_tr = 0
    sid_te = 0
    for i in range(4):
        eid_tr = sid_tr + num_per_file_tr if i != 3 else num_tr
        eid_te = sid_te + num_per_file_te if i != 3 else num_te
        train_x_i, train_y_i = data_tr[0][sid_tr: eid_tr], data_tr[1][sid_tr: eid_tr]
        test_x_i, test_y_i = data_te[0][sid_te: eid_te], data_te[1][sid_te: eid_te]
        dataset_i = {'train': [train_x_i, train_y_i], 'test': [test_x_i, test_y_i]}
        np.save(saved_path+str(i)+'.npy', dataset_i)
        sid_tr, sid_te = eid_tr, eid_te

    # convert labels to one-hot encoding
    dataset = convert_to_onehot(dataset)
    et = time.time()
    print("Time taken for reading files: {:.4f}s".format(et - st))
    return dataset

def _AWA_read_img_from_file(data_dir, file_name, img_height, img_width):
    count = 0
    imgs = []
    labels = []

    with open(file_name) as f:
        for line in f:
            img_name, img_label = line.split()
            img_file = data_dir.rstrip('\/') + '/' + img_name
            img = cv2.imread(img_file).astype(np.float32)

            # HWC -> WHC, compatible with caffe weights
            img = cv2.resize(img, (img_width, img_height))

            # convert RGB to BGR
            img_r, img_g, img_b = np.split(img, 3, axis=2)
            img = np.concatenate((img_b, img_g, img_r), axis=2)

            # extract mean
            img -= IMG_MEAN

            imgs += [img]
            labels += [int(img_label)]
            count += 1

            if count % 1000 == 0:
                print("\tFinish reading {:07d}".format(count))

    return np.array(imgs), np.array(labels)

def _CUB_read_img_from_file(data_dir, file_name, img_height, img_width):
    count = 0
    imgs = []
    labels = []

    with open(file_name) as f:
        for line in f:
            img_name, img_label = line.split()
            img_file = data_dir.rstrip('\/') + '/' + img_name
            img = cv2.imread(img_file).astype(np.float32)

            # HWC -> WHC, compatible with caffe weights
            img = cv2.resize(img, (img_width, img_height))

            # convert RGB to BGR
            img_r, img_g, img_b = np.split(img, 3, axis=2)
            img = np.concatenate((img_b, img_g, img_r), axis=2)

            # extract mean
            img -= IMG_MEAN

            imgs += [img]
            labels += [int(img_label)]
            count += 1

            if count % 1000 == 0:
                print("\tFinish reading {:07d}".format(count))

    # convert the labels to one-hot encoding (not now to reduce size)
    # y = dense_to_one_hot(np.array(labels))
    return np.array(imgs), np.array(labels)



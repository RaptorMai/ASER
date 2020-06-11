import numpy as np
from copy import deepcopy
import pickle
import gzip
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from collections import Counter
from operator import itemgetter

class imageSet(object):
    def __init__(self, image, label):
        self.images = image
        self.labels = label


class readMNIST(object):
    def __init__(self, datapath='mnist.pkl.gz'):
        self.data_path = datapath

    def get_dataset(self):
        f = gzip.open(self.data_path, 'rb')
        self.data_set = pickle.load(f, encoding='iso-8859-1')
        f.close()
        return self.data_set


class Data(object):
    def __init__(self, x, y, full=False, color=False):
        if color:
            self.x = x / 255.0
            self.next_x = x /255.0
        else:
            self.x = x
            self.next_x = x
        self.y = y
        self.full = full
        self.next_y = y

    def get_dims(self):
        # Get data input and output dimensions
        print("input size {}\noutput size {}".format(self.x.shape[1], self.y.shape[1]))
        return self.x.shape[1], self.y.shape[1]

    def show_sample(self, ):
        idx = np.random.choice(self.x.shape[0])
        plt.subplot(1, 2, 1)
        if self.x[0].shape[2] == 1:
            plt.imshow( np.squeeze(self.x[0]))
        else:
            plt.imshow(self.x[0])
        plt.title("original task image")
        plt.subplot(1, 2, 2)
        if self.x[0].shape[2] == 1:
            plt.imshow( np.squeeze(self.next_x[0]))
        else:
            plt.imshow(self.next_x[0])
        plt.title(self.get_name())
        plt.axis('off')
        plt.show()

    def create_output(self):
        if self.full:
            ret = imageSet(self.next_x.reshape((-1, self.x.shape[1] ** 2)), self.next_y)
        else:
            ret = imageSet(self.next_x, self.next_y)
        return ret

    @staticmethod
    def clip_minmax(l, min_=0., max_=1.):
        return np.clip(l, min_, max_)

    def get_name(self):
        if hasattr(self, 'factor'):
            return str(self.__class__.__name__) + '_' + str(self.factor)

    def next_task(self, *args):
        self.next_x = self.x
        self.next_y = self.y
        return self.create_output()


class Noisy(Data):
    def __init__(self, x, y, full=False, color=False):
        super(Noisy, self).__init__(x, y, full, color)

    def next_task(self, noise_factor=0.8, sig=0.1, noise_type='Gaussian'):
        next_x = deepcopy(self.x)
        self.factor = noise_factor
        if noise_type == 'Gaussian':
            self.next_x = next_x + noise_factor * np.random.normal(loc=0.0, scale=sig, size=next_x.shape)
        elif noise_factor == 'S&P':
            # TODO implement S&P
            pass

        self.next_x = super().clip_minmax(self.next_x, 0, 1)

        return super().create_output()


class Blurring(Data):
    def __init__(self, x, y, full=False, color=False):
        super(Blurring, self).__init__(x, y, full, color)

    def next_task(self, blurry_factor=0.6, blurry_type='Gaussian'):
        next_x = deepcopy(self.x)
        self.factor = blurry_factor
        if blurry_type == 'Gaussian':
            self.next_x = gaussian_filter(next_x, sigma=blurry_factor)
        elif blurry_type == 'Average':
            pass
            # TODO implement average

        self.next_x = super().clip_minmax(self.next_x, 0, 1)

        return super().create_output()


class Occlusion(Data):
    def __init__(self, x, y, full=False, color=False):
        super(Occlusion, self).__init__(x, y, full, color)

    def next_task(self, occlusion_factor=0.2, random_seed=0):
        next_x = deepcopy(self.x)
        self.factor = occlusion_factor
        self.image_size = next_x.shape[1]
        random.seed(random_seed)

        occlusion_size = int(occlusion_factor * self.image_size)
        half_size = occlusion_size // 2
        occlusion_x = random.randint(min(half_size, self.image_size - half_size),
                                     max(half_size, self.image_size - half_size))
        occlusion_y = random.randint(min(half_size, self.image_size - half_size),
                                     max(half_size, self.image_size - half_size))

        # self.next_x = next_x.reshape((-1, self.image_size, self.image_size))

        next_x[:, max((occlusion_x - half_size), 0):min((occlusion_x + half_size), self.image_size), \
        max((occlusion_y - half_size), 0):min((occlusion_y + half_size), self.image_size)] = 1

        self.next_x = next_x
        super().clip_minmax(self.next_x, 0, 1)

        return super().create_output()


class Imbalance_step(Data):
    def __init__(self, x, y, full=False,color=False):
        super(Imbalance_step, self).__init__(x, y, full, color)

    def next_task(self, random_seed=0, factor=(10, 0.5)):
        np.random.seed(random_seed)
        p, u = factor
        total_classes = self.y.shape[1]
        minor_cnt = (self.x.shape[0] // total_classes) // p
        minor = np.random.choice(total_classes, int(total_classes * u), replace=False)
        cnt = {i: 0 for i in minor}
        y = np.argmax(self.y, axis=1)
        idx_keep = []
        self.label_keep = []
        for i, label in enumerate(y):
            if label not in cnt:
                self.label_keep.append(label)
                idx_keep.append(i)
            elif cnt[label] < minor_cnt:
                self.label_keep.append(label)
                idx_keep.append(i)
                cnt[label] += 1
        cat_y = y[idx_keep]
        new_y = np.eye(10)[cat_y]
        new_x = self.x[idx_keep]
        return imageSet(new_x, new_y)

    def show_distribution(self, label, ax, title):
        c = list(Counter(self.label_keep).items())
        c.sort(key=itemgetter(1))
        labels, values = zip(*c)
        indexes = np.arange(len(labels))
        width = 0.5

        ax.bar(indexes, values, width, label=label)
        ax.set_xticks(indexes + width * 0.5, minor=False)
        ax.set_xticklabels(labels, fontdict=None, minor=False)
        ax.title.set_text(title)
        #ax.xticks(indexes + width * 0.5, labels)
        # plt.hist(self.label_keep, label=label)
        ax.legend()

class Imbalance_linear(Data):
    def __init__(self, x, y, full=False, color=False):
        super(Imbalance_linear, self).__init__(x, y, full, color)

    def next_task(self, random_seed=0, factor=(10)):
        np.random.seed(random_seed)
        p = factor[0]
        total_classes = self.y.shape[1]
        xPclass = self.x.shape[0] // total_classes
        minor_cnt = xPclass // p
        minor_cnt_list = np.arange(minor_cnt, xPclass+1, (xPclass-minor_cnt)//(total_classes-1))
        class_idx = np.arange(0, total_classes)
        np.random.shuffle(class_idx)
        cnt_ref = {class_idx[i]: minor_cnt_list[i] for i in range(total_classes)}
        cnt = {i: 0 for i in class_idx}
        y = np.argmax(self.y, axis=1)
        idx_keep = []
        self.label_keep = []
        for i, label in enumerate(y):
            if cnt[label] < cnt_ref[label]:
                self.label_keep.append(label)
                idx_keep.append(i)
                cnt[label] += 1
        cat_y = y[idx_keep]
        new_y = np.eye(10)[cat_y]
        new_x = self.x[idx_keep]
        return imageSet(new_x, new_y)

    def show_distribution(self, label, ax, title):
        c = list(Counter(self.label_keep).items())
        c.sort(key=itemgetter(1))
        labels, values = zip(*c)
        indexes = np.arange(len(labels))
        width = 0.5

        ax.bar(indexes, values, width, label=label)
        ax.set_xticks(indexes + width * 0.5, minor=False)
        ax.title.set_text(title)
        #ax.xticks(indexes + width * 0.5, labels)
        # plt.hist(self.label_keep, label=label)
        ax.legend()

class UnevenLabels(Data):
    def __init__(self, x, y, full=False):
        super(UnevenLabels, self).__init__(x, y, full)

    def next_task(self, mean_var_list):
        pass
        # import scipy.stats as stat
        # # mean_var_list is a list of tuples [(m1, v1), (m2, v2),...]
        # # For now, accept only the number of distributions that can divide the total number of
        # # sample size (in the case of MNIST, 50k)
        # num_dist = len(mean_var_list)
        # assert(self.X_train.shape[0] % num_dist == 0)
        # size_dist = self.X_train.shape[0] // num_dist
        #
        # # next_x_train = deepcopy(self.X_train)
        # # next_x_test = deepcopy(self.X_test)
        # # next_x_val = deepcopy(self.X_val)
        #
        # class_labels = set(self.Y_train)        # assuming self.Y_train is not one-hot encoded yet
        #
        # # empty arrays where data is stacked
        # x_res = np.empty([0, self.image_size])
        # y_res = []
        # for m, v in mean_var_list:
        #     temp_norm = stat.norm(m, v)
        #     prob_labels = temp_norm.pdf(class_labels)
        #     sample_prob = prob_labels / np.sum(prob_labels)
        #     max_prob = np.max(sample_prob)
        #     idx_to_keep = [i for i, label in enumerate(self.Y_train)
        #                    if random.uniform(0, max_prob) < sample_probs[label]]
        #     idx_to_keep = np.random.choice(idx_to_keep, size=size_dist, replace=False)
        #     x_res = np.vstack([x_res, self.X_train[idx_to_keep]])
        #     y_res += list(self.Y_train[idx_to_keep])
        # y_res = np.array(y_res)

class Brighter(Data):
    def __init__(self, x, y, color=True, full=False):
        super(Brighter, self).__init__(x, y, color, full)


    def next_task(self, brightness_factor=0.5):
        pass
        #not using now
        # next_x_train = deepcopy(self.X_train)
        # next_x_test = deepcopy(self.X_test)
        # next_x_val = deepcopy(self.X_val)
        #
        # self.next_x_train = next_x_train + brightness_factor
        # self.next_x_val = next_x_val + brightness_factor
        # self.next_x_test = next_x_test + brightness_factor
        #
        # super().clip_minmax([self.next_x_train, self.next_x_val, self.next_x_test])
        #
        # return super().create_output()


class Darker(Data):
    def __init__(self, x, y, color=True, full=False):
        super(Darker, self).__init__(x, y, color, full)


    def next_task(self, brightness_factor=0.5):
        pass
        # not using it npow
        # next_x_train = deepcopy(self.X_train)
        # next_x_test = deepcopy(self.X_test)
        # next_x_val = deepcopy(self.X_val)
        #
        # self.next_x_train = next_x_train - brightness_factor
        # self.next_x_val = next_x_val - brightness_factor
        # self.next_x_test = next_x_test - brightness_factor
        #
        # super().clip_minmax([self.next_x_train, self.next_x_val, self.next_x_test])
        #
        #
        # return super().create_output()


class Permuted(Data):
    def __init__(self, x, y, color=True, full=False):
        super(Permuted, self).__init__(x, y,  color, full)


    def next_task(self, seed=None, *args):
        pass
        # not using permuted now
        # next_x_train = deepcopy(self.X_train)
        # next_x_test = deepcopy(self.X_test)
        # next_x_val = deepcopy(self.X_val)
        #
        # if seed:
        #     np.random.seed(seed)
        # else:
        #     np.random.seed(np.random.randint(0, 2**32 - 1))
        # perm_inds = list(range(self.X_train.shape[1]))
        # np.random.shuffle(perm_inds)
        #
        # # Retrieve train data
        # self.next_x_train = next_x_train[:, perm_inds]
        # self.next_x_val = next_x_val[:, perm_inds]
        # # Retrieve test data
        # self.next_x_test = next_x_test[:, perm_inds]
        #
        #
        # return super().create_output()



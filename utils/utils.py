import os, sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from Data.data import *
import numpy as np
import utils.global_vars as global_vars


import tensorflow as tf

def weight_variable(shape, std=0.1):
    initial = tf.truncated_normal(shape, stddev=std)
    return tf.Variable(initial, name='weights')


def bias_variable(shape, const=0.1):
    initial = tf.constant(const, shape=shape)
    return tf.Variable(initial, name='biases')



def generate_mnist_tasks(dataset, params):
    instance = {"Permuted": Permuted(dataset),
                "Occlusion": Occlusion(dataset),
                "Darker": Darker(dataset),
                "Brighter": Brighter(dataset),
                "Blurring": Blurring(dataset),
                "Noisy": Noisy(dataset),
                "Original": Data(dataset)}

    dataset = [[] for _ in range(3)]
    for i in params:
        if i[1] is None:
            train, val, test = instance[i[0]].next_task()
        else:
            train, val, test = instance[i[0]].next_task(i[1])
        dataset[0].append(train)
        dataset[1].append(val)
        dataset[2].append(test)

    return dataset, [instance[i[0]] for i in params]


def update_reservior(current_image, current_label, episodic_images, episodic_labels, M, N):
    """
    Update the episodic memory with current example using the reservior sampling
    """
    if M > N:
        episodic_images[N] = current_image
        episodic_labels[N] = current_label
    else:
        j = np.random.randint(0, N)
        if j < M:
           episodic_images[j] = current_image
           episodic_labels[j] = current_label


def store_results(model_name, detail, **kwargs):
    for name, data in kwargs.items():
        file_name = 'test_result/' + model_name +'/' +name +'_' + detail
        print(file_name)
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
    return file_name

def resize_batch(x_batch, shape=(10, 224, 224, 3), order=0):
    k = 0
    x_resized = np.zeros(shape)
    for x in x_batch:
        x_resized[k] = resize(x, shape[1:], order=order)
        k += 1
    return x_resized

def acc_store(model, valset, task_id, sess, store, index, full_batch=True):
    if model.name == 'EWC':
        logit_mask = np.zeros(model.out_dim)
    else:
        logit_mask = np.zeros([model.n_tasks, model.out_dim])
    print('Training task {}'.format(task_id))
    for task in range(model.n_tasks):
        if 'mnist' in model.args.data:
            feed_dict = {model.x: valset[task].images, model.y_: valset[task].labels}
            tmp_acc = sess.run(model.accuracy, feed_dict=feed_dict)

        elif model.args.arch in global_vars.CONV_ARCH:

            tmp_acc = []
            single_acc = []
            task_size = valset[task].images.shape[0]

            # stochastic estimation of test accuracy
            if model.args.data in global_vars.MINIB_TEST:
                test_batch_size = 512
                total_batch = int(np.ceil(task_size * 1.0 / test_batch_size))
                perm_inds = list(range(task_size))
                np.random.shuffle(perm_inds)
                cur_x_train = valset[task].images[perm_inds]
                cur_y_train = valset[task].labels[perm_inds]
                for i in range(total_batch):
                    start_ind = i * test_batch_size
                    end_ind = np.min([(i + 1) * test_batch_size, task_size])
                    batch_x = cur_x_train[start_ind:end_ind, :]
                    batch_y = cur_y_train[start_ind:end_ind, :]
                    feed_dict = {model.x: batch_x, model.y_: batch_y}
                    feed_dict[model.train_phase] = False
                    if model.head == 'single':
                        tmp_acc.append(model.accuracy.eval(feed_dict=feed_dict))
                    elif model.head == 'multi':
                        logit_mask[:] = 0
                        logit_mask[task][model.task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        tmp_acc.append(model.accuracy[task].eval(feed_dict=feed_dict))
            elif model.args.data in global_vars.BATCH_TEST:
                # Full-batch test accuracy
                if model.head == 'multi':
                    x = valset[task].images
                    feed_dict = {model.x: x, model.y_: valset[task].labels, model.train_phase: False}
                    if model.name == 'EWC':
                        logit_mask[:] = 0
                        logit_mask[model.task_labels[task]] = 1.0
                        feed_dict[model.keep_prob] = 1.0
                        feed_dict[model.output_mask] = logit_mask
                        prediction = model.pruned_logits.eval(feed_dict=feed_dict)
                        y_ = valset[task].labels
                        tmp_acc = model.accuracy.eval(feed_dict=feed_dict)
                    else:
                        logit_mask[:] = 0
                        logit_mask[task][model.task_labels[task]] = 1.0
                        logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(model.output_mask, logit_mask)}
                        feed_dict.update(logit_mask_dict)
                        tmp_acc = model.accuracy[task].eval(feed_dict=feed_dict)

                elif model.head == 'single':
                    x = valset[task].images
                    feed_dict = {model.x: x, model.y_: valset[task].labels}
                    if 'resnet' in model.arch:
                        feed_dict[model.train_phase] = False
                    tmp_acc = sess.run(model.accuracy, feed_dict=feed_dict)
            else:
                raise Exception('wrong testing')

        if index is not None:
            store[task][task_id, index] = np.mean(tmp_acc)
        else:
            store[task][task_id] = np.mean(tmp_acc)
        print("{}: {:.2f} ".format(task, np.mean(tmp_acc)), end="\t")
    print()


def test_init_same_task_size(model, trainset, batch_size, no_epochs):
    test_accs = []
    end_task_accs = []
    end_epoch_accs = []
    #TODO
    #train_grad = np.zeros(result_size)
    # each task has one ndarray with size total_batch * no_epochs
    task_size = max([trainset[i].images.shape[0] for i in range(model.n_tasks)])
    # Total batch for one epoch
    total_batch = int(np.ceil(task_size * 1.0 / batch_size))
    for task in range(model.n_tasks):
        result_size = (model.n_tasks, int(np.ceil(total_batch * no_epochs / model.disp_freq_train)))
        test_accs.append(np.zeros(result_size))
        end_task_accs.append(np.zeros(model.n_tasks))
        end_epoch_accs.append(np.zeros((model.n_tasks, model.no_epochs)))
    train_accs = deepcopy(test_accs)
    train_end_task_accs = deepcopy(end_task_accs)
    return train_accs, train_end_task_accs, test_accs, end_task_accs, end_epoch_accs



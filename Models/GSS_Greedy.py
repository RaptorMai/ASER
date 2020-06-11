import numpy as np
import math
from copy import deepcopy
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import utils.utils as utils
from utils.buffer import Buffer
from Models.Tiny_mem import Tiny_mem
from utils.buffer import GSS_Buffer
from Models.model_wrapper import CLInterface
from Metrics.metrics import avg_acc
from utils.build_architecture import fully_model, resnet18, resnet_params
import utils.global_vars as global_vars
from utils.resnet_utils import load_resnet18

class GSS_Greedy(Tiny_mem):
    def __init__(self,
                 input_size,
                 output_size,
                 learning_rate=0.1,
                 disp_freq_train=200,
                 optimizer=tf.train.GradientDescentOptimizer,
                 arch='fully',
                 head='single',
                 scheme='indep',
                 extra_arg=None,
                 is_ATT_DATASET=False,
                 attr=None,
                 **kwargs
                 ):
        super(GSS_Greedy, self).__init__(input_size=input_size,
                 output_size=output_size,
                 learning_rate=learning_rate,
                 disp_freq_train=disp_freq_train,
                 optimizer=optimizer,
                 arch=arch,
                 head=head,
                 scheme=scheme,
                 extra_arg=extra_arg,
                 is_ATT_DATASET=is_ATT_DATASET,
                 attr=attr,
                 **kwargs)


    #################################################################################
    #### External APIs of the class. These will be called/ exposed externally #######
    #################################################################################

    def train_model(self, trainset, valset, sess, no_epochs, batch_size, task_labels=None):
        self.no_epochs = no_epochs
        self.batch_size = batch_size
        #self._define_mem(self.n_tasks)
        self.task_labels = task_labels

        self.init_vars()
        self.weights_store_ops()

        if self.args.data not in global_vars.SPLIT_DATA:
            episodic_mem_size = self.mem_size * self.out_dim * self.n_tasks
        else:
            episodic_mem_size = self.mem_size * self.out_dim

        gss_buffer = GSS_Buffer(self.args, episodic_mem_size, self.in_dim, self.out_dim, self.eps_mem_batch)

        train_accs, train_end_task_accs, test_accs, end_task_accs, end_epoch_accs = utils.test_init_same_task_size(self, trainset, batch_size, no_epochs)

        if self.args.data in global_vars.SPLIT_DATA:
            # List to store the classes that we have so far - used at test time
            test_labels = []

        # Build additional computational graph
        self._train_step()
        self._define_acc()

        # init sess
        sess.run(tf.global_variables_initializer())
        if self.pretrained:
            # below code imports pretrained parameters to the model (resnet18)
            sess.run(self.load_pretrained_ops)

        for task_id in range(self.n_tasks):
            if self.args.data in global_vars.SPLIT_DATA:
                # Test for the tasks that we've seen so far
                # not using now
                test_labels += task_labels[task_id]

            task_size = trainset[task_id].images.shape[0]
            total_batch = int(np.ceil(task_size * 1.0 / batch_size))
            print('Received {} images, {} labels at task {}'.format(task_size, trainset[task_id].labels.shape[0],
                                                                    task_id))
            print('Unique labels in the task: {}'.format(np.unique(np.nonzero(trainset[task_id].labels)[1])))
            print('Total batch: {}'.format(total_batch))
            for epoch in range(no_epochs):
                perm_inds = list(range(task_size))
                np.random.shuffle(perm_inds)
                cur_x_train = trainset[task_id].images[perm_inds]
                cur_y_train = trainset[task_id].labels[perm_inds]

                for i in range(total_batch):
                    start_ind = i * batch_size
                    end_ind = np.min([(i+1) * batch_size, task_size])
                    batch_x = cur_x_train[start_ind:end_ind, :]
                    batch_y = cur_y_train[start_ind:end_ind, :]

                    for iter in range(self.args.num_iter_batch):
                        if self.all_data:
                            feed_dict = {self.x: batch_x, self.y_: batch_y}
                            if 'resnet' in self.arch:
                                feed_dict[self.train_phase] = True
                            _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
                        else:
                            if task_id > 0:
                                mem_images, mem_labels = gss_buffer.get_mem(self, sess, batch_x, batch_y, iter)

                                # Train on a batch of episodic memory first
                                er_train_x_batch = np.concatenate((mem_images, batch_x), axis=0)
                                er_train_y_batch = np.concatenate((mem_labels, batch_y), axis=0)
                            else:
                                er_train_x_batch = batch_x
                                er_train_y_batch = batch_y

                            if self.separate:
                                #train batch
                                feed_dict = {self.x: batch_x, self.y_: batch_y}
                                if 'resnet' in self.arch:
                                    feed_dict[self.train_phase] = True

                                _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)

                                #train mem
                                if task_id >0:
                                    feed_dict = {self.x: mem_images, self.y_: mem_labels}
                                    if 'resnet' in self.arch:
                                        feed_dict[self.train_phase] = True
                                    _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)

                                #TODO add incre and indep version
                            else:
                                feed_dict = {self.x: er_train_x_batch, self.y_: er_train_y_batch}

                                if 'resnet' in self.arch:
                                    feed_dict[self.train_phase] = True

                                # memory mask
                                if self.head == 'multi' or self.scheme == 'incre' or self.scheme == 'indep':
                                    #multi head
                                    self.nd_logit_mask[:] = 0
                                    for tt in range(task_id + 1):
                                        self.nd_logit_mask[tt][task_labels[tt]] = 1.0
                                    logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(self.output_mask, self.nd_logit_mask)}
                                    feed_dict.update(logit_mask_dict)
                                    feed_dict[self.mem_batch_size] = float(er_train_x_batch.shape[0])

                                #g_and_v, _, loss = sess.run([self.grads_and_vars, self.train_step, self.loss],feed_dict=feed_dict)
                                _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)

                    # Reservoir update
                    # for er_x, er_y_ in zip(batch_x, batch_y):
                    #     utils.update_reservior(er_x, er_y_, self.episodic_images, self.episodic_labels, self.episodic_mem_size, self.examples_seen_so_far)
                    #     self.examples_seen_so_far += 1
                    gss_buffer.update_mem(sess, self, batch_x, batch_y, task_id)

                    if (math.isnan(loss)):
                        print('ERROR: NaNs NaNs NaNs!!!')
                        sys.exit(0)

                    if (epoch * total_batch + i) % self.disp_freq_train == 0:
                        # Compute the mean gradient of all trainable parameters
                        print('Step {}: loss {}'.format((epoch * total_batch + i), loss))
                        #TODO add train_grad
                        #train_grad[task_id, int((epoch * total_batch + i) / self.disp_freq_train)] = self.grad_mean(g_and_v)
                        index = int((epoch * total_batch + i) / self.disp_freq_train)
                        # training acc is not efficient using the current function
                        #utils.acc_store(self, trainset, task_id, sess, train_accs, index)
                        utils.acc_store(self, valset, task_id, sess, test_accs, index)

                print('done epoch {} for task {}'.format(epoch, task_id))
                print(loss)
                utils.acc_store(self, valset, task_id, sess, end_epoch_accs, None)
            #utils.acc_store(self, trainset, task_id, sess, train_end_task_accs, None)
            print('done task {}'.format(task_id))
            print(loss)
            utils.acc_store(self, valset, task_id, sess, end_task_accs, None)



        avg_acc_test = avg_acc(end_task_accs)
        #avg_acc_train = avg_acc(train_end_task_accs)
        #print("average train accuracy")
        #print(avg_acc_train)
        print("average test accuracy")
        print(avg_acc_test)
        ### store
        # utils.store_results("GSS_Greedy", self.model_name, test_acc=test_accs)
        # utils.store_results("GSS_Greedy", self.model_name, end_task_accs=end_task_accs)
        # utils.store_results("GSS_Greedy", self.model_name, avg_acc_=avg_acc_test)
        # utils.store_results("GSS_Greedy", self.model_name, train_grad=train_grad)

        # utils.store_results("GSS_Greedy", self.model_name, train_accs=train_accs)
        # utils.store_results("GSS_Greedy", self.model_name, train_end_task_accs=train_end_task_accs)
        # utils.store_results("GSS_Greedy", self.model_name, avg_acc_train=avg_acc_train)
        return end_task_accs, avg_acc_test



    ####################################################################################
    #### Internal APIs of the class. These should not be called/ exposed externally ####
    ####################################################################################
    def _is_properly_setup(self):
        super(GSS_Greedy, self)._is_properly_setup()
        if self.arch == 'fully' and len(self.in_dim) != 1:
            raise Exception("input size should have length 1 for fully connected ")
        elif 'resnet' in self.arch and len(self.in_dim) != 3:
            raise Exception("input size should have length 3 for CNN ")

    def _train_step(self):
        if self.scheme == 'full':
            with tf.variable_scope('vanilla_loss'):

                if self.multi_label:
                    self.loss = tf.reduce_mean(
                        tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.y),
                                      axis=1))
                else:
                    self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y)
                    self.loss = tf.reduce_mean(self.cross_entropy)

            # Compute the gradients of regularized loss
            self.reg_gradients_vars = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_vars)

            # Define training operation
            self.train_step = self.optimizer.apply_gradients(self.reg_gradients_vars)

        elif self.scheme == 'indep' or self.scheme == 'incre' or self.head == 'multi':
            self.output_mask = [tf.placeholder(dtype=tf.float32, shape=[self.out_dim]) for i in range(self.n_tasks)]
            self.task_pruned_logits = []
            self.unweighted_entropy = []
            for i in range(self.n_tasks):
                self.task_pruned_logits.append(
                    tf.where(tf.tile(tf.equal(self.output_mask[i][None, :], 1.0), [tf.shape(self.y)[0], 1]), self.y,
                             global_vars.NEG_INF * tf.ones_like(self.y)))
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.task_pruned_logits[i])
                adjusted_entropy = tf.reduce_sum(
                    tf.cast(tf.tile(tf.equal(self.output_mask[i][None, :], 1.0), [tf.shape(self.y_)[0], 1]),
                            dtype=tf.float32) * self.y_, axis=1) * cross_entropy
                self.unweighted_entropy.append(tf.reduce_sum(adjusted_entropy))  # We will average it later on

            # Create operations for loss and gradient calculation
            with tf.variable_scope('vanilla_loss'):
                self.loss = tf.add_n([self.unweighted_entropy[i] for i in range(self.n_tasks)]) / self.mem_batch_size
            self.reg_gradients_vars = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_vars)

            # Define training operation
            self.train_step = self.optimizer.apply_gradients(self.reg_gradients_vars)

        # for gradient based sampling - find vectorized gradients
        sample_gradients = tf.gradients(self.loss, self.trainable_vars)
        self.vectorized_gradients = tf.concat([tf.reshape(g, [-1]) for g in sample_gradients], axis=0)

    def _assign_pretrained_ops(self):
        """
        Given the pre-trained weights and the mapping from pytorch variable names to tf variable names, assign the
        weights to the model parameters.
        :param sess:            (tf.Session)
        :param saved_param:     (dict) Key: pytorch variable names, value: pretrained weights
        :param converter:       (dict) Key: tf variable names,      value: pytorch variable names
        """
        saved_param, converter = load_resnet18()
        load_pretrained_ops = []
        for var in self.trainable_vars:
            tf_name = var.name
            torch_name = converter[tf_name]
            w = tf.convert_to_tensor(saved_param[torch_name], dtype=tf.float32)
            load_pretrained_ops.append(tf.assign(var, w))
        self.load_pretrained_ops = tf.group(*load_pretrained_ops)

    @property
    def model_name(self):
        if self.args.data == 'mnist':
            return "GSS_Greedy_data{}_arch{}_epoch{}_batch{}_optmz{}_lr{}_memSize{}_memSizeBatch{}_dis{}_numtasks{}_pretrain{}_scheme{}_num_sample_grad{}_num_iter{}".\
                format(self.args.data, self.arch, self.no_epochs, self.batch_size, self.args.optimizer,
                       self.learning_rate, self.mem_size, self.eps_mem_batch, self.disp_freq_train, self.n_tasks, self.pretrained, self.scheme, self.args.num_sample_grad, self.args.num_iter_batch)
        else:
            return "GSS_Greedy_data{}_arch{}_epoch{}_batch{}_optmz{}_lr{}_memSize{}_memSizeBatch{}_dis{}_numtasks{}_pretrain{}_scheme{}_num_sample_grad{}_num_iter{}". \
                format(self.args.data, self.arch, self.no_epochs, self.batch_size, self.args.optimizer,
                       self.learning_rate, self.mem_size, self.eps_mem_batch, self.disp_freq_train, self.n_tasks, self.pretrained, self.scheme, self.args.num_sample_grad, self.args.num_iter_batch)

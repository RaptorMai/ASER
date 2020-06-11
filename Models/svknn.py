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
from utils.sv_knn_buffer import SVKNNBuffer
from Models.model_wrapper import CLInterface
from Metrics.metrics import avg_acc
from utils.build_architecture import fully_model, resnet18, resnet_params
import utils.global_vars as global_vars


class SVKNN(Tiny_mem):
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
        super(SVKNN, self).__init__(input_size=input_size,
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
        print(self.model_name)
        episodic_mem_size = self.mem_size * self.out_dim
        np.random.seed(self.args.random_seed)
        sv_knn_buffer = SVKNNBuffer(self.args, episodic_mem_size, self.in_dim, self.out_dim, self.eps_mem_batch)

        train_accs, train_end_task_accs, test_accs, end_task_accs, end_epoch_accs = utils.test_init_same_task_size(self, trainset, batch_size, no_epochs)

        if self.args.data in global_vars.SPLIT_DATA:
            # List to store the classes that we have so far - used at test time
            test_labels = []

        # Build additional computational graph
        self._train_step()
        self._define_acc()

        # init sess
        sess.run(tf.global_variables_initializer())
        self.init_vars()
        self.weights_store_ops()
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
                                # mem_images, mem_labels = sv_knn_buffer.get_mem(self, sess, current_x=None, current_y=None, exclude=None)
                                if self.args.is_sv_retrieval and not self.args.is_offline:
                                    mem_images, mem_labels = sv_knn_buffer.get_sv_mem(self, sess, batch_x, batch_y)
                                elif self.args.is_mir:
                                    mem_images, mem_labels = sv_knn_buffer.get_mem(self, sess, current_x=batch_x, current_y=batch_y, exclude=task_id)
                                elif self.args.is_dist_retrieval:
                                    mem_images, mem_labels = sv_knn_buffer.get_dist_mem(self, sess, current_x=batch_x, current_y=batch_y)
                                else:
                                    mem_images, mem_labels = sv_knn_buffer.get_mem(self, sess, current_x=None, current_y=None, exclude=task_id)

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


                    if self.args.is_sv_update:
                        sv_knn_buffer.update_sv_mem(batch_x, batch_y, task_id, sess, self)
                    else:
                        sv_knn_buffer.update_mem(batch_x, batch_y, task_id)
                    #self.print_task_ids_freq_mem_samples(sv_knn_buffer, i)

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

        return end_task_accs, avg_acc_test



    ####################################################################################
    #### Internal APIs of the class. These should not be called/ exposed externally ####
    ####################################################################################
    def _is_properly_setup(self):
        super(SVKNN, self)._is_properly_setup()
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

    @property
    def model_name(self):
        if self.args.data == 'mnist':
            base = "SVKNN_data{}_arch{}_epoch{}_batch{}_optmz{}_lr{}_memSize{}_memBatch{}_dis{}_ntasks{}_n_sam{}_k{}_n_runs{}_SVretv{}_SVup{}".format(
                self.args.data, self.arch, self.no_epochs, self.batch_size, self.args.optimizer,
                self.learning_rate, self.mem_size, self.eps_mem_batch, self.disp_freq_train, self.n_tasks,
                self.args.max_num_samples, self.args.num_k, self.args.num_runs, int(self.args.is_sv_retrieval),
                int(self.args.is_sv_update))
            if self.args.is_penalized_curr_task:
                base += '_penal' + str(self.args.is_penalized_curr_task)
            if self.args.is_offline:
                base += '_off' + str(self.args.is_offline)
            if self.args.is_adversarial_sv:
                base += '_adv' + str(self.args.is_adversarial_sv) + '_atype' +str(self.args.adversarial_type) + '_advCo' + str(self.args.adv_coeff)
            if self.args.is_accum_curr:
                base += '_accum' + str(self.args.is_accum_curr)
            if self.args.is_mir:
                base += '_mir'
            if self.args.is_dist_retrieval:
                base += '_disRetr'
            return base
        else:
            base = "SVKNN_data{}_arch{}_epoch{}_batch{}_optmz{}_lr{}_memSize{}_memBatch{}_dis{}_ntasks{}_n_sam{}_k{}_n_runs{}_SVretv{}_SVup{}".format(self.args.data, self.arch, self.no_epochs, self.batch_size, self.args.optimizer,
                       self.learning_rate, self.mem_size, self.eps_mem_batch, self.disp_freq_train, self.n_tasks,
                       self.args.max_num_samples, self.args.num_k, self.args.num_runs, int(self.args.is_sv_retrieval), int(self.args.is_sv_update))
            if self.args.is_penalized_curr_task:
                base += '_penal' + str(self.args.is_penalized_curr_task)
            if self.args.is_offline:
                base += '_off' + str(self.args.is_offline)
            if self.args.is_adversarial_sv:
                base += '_adv' + str(self.args.is_adversarial_sv) + '_atype' +str(self.args.adversarial_type) + '_advCo' + str(self.args.adv_coeff)
            if self.args.is_accum_curr:
                base += '_accum' + str(self.args.is_accum_curr)
            if self.args.is_mir:
                base += '_ismir'
            if self.args.is_dist_retrieval:
                base += '_disRetr'
            return base


    def print_task_ids_freq_mem_samples(self, buffer, i):
        if (i % 50 == 0):
            unique, count = np.unique(buffer.episodic_labels_int, return_counts=True)
            freq = np.asarray((unique, count), dtype=np.int)
            print(freq)
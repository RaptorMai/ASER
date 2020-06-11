import tensorflow as tf
import numpy as np
import os, sys
from copy import deepcopy
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import utils.utils as utils
from Models.model_wrapper import CLInterface
from Metrics.metrics import avg_acc
from utils.build_architecture import fully_model, resnet18, resnet_params
import time
import math
import utils.global_vars as global_vars
from utils.resnet_utils import load_resnet18



class EWC(CLInterface):
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

        super(EWC, self).__init__()
        if arch not in global_vars.VALID_ARCH:
            raise Exception('not valid architecture')
        else:
            self.arch = arch

        self.pretrained = extra_arg.pretrained
        self.in_dim = input_size
        self.out_dim = output_size
        if self.pretrained:
            self.x = tf.placeholder(tf.float32, shape=[None] + [224, 224, 3])   # model pretrained on ImageNet dataset
        else:
            self.x = tf.placeholder(tf.float32, shape=[None] + input_size)
        self.y_ = tf.placeholder(tf.float32, shape=[None, output_size])
        self.learning_rate = learning_rate
        self.disp_freq_train = disp_freq_train
        self.head = head
        self.scheme = scheme
        self.optimizer = optimizer(learning_rate)
        self.n_tasks = extra_arg.num_tasks
        self.all_data = extra_arg.all_data
        self.separate = extra_arg.seperate
        self.args = extra_arg
        self.trainable_vars = []

        #according to author of AGEM, this is true only when we use CUB dataset
        self.is_ATT_DATASET = is_ATT_DATASET
        # Class attributes for zero shot transfer
        self.class_attr = attr
        self.args = extra_arg

        #EWC
        # self.fisher_num_samples = extra_arg.fisher_num_samples
        self.lamda = tf.constant(extra_arg.lambda_, shape=[1], dtype=tf.float32)
        self.fisher_ema_decay = extra_arg.fisher_ema_decay
        self.fisher_update_after = extra_arg.fisher_update_after

        if self.args.arch in global_vars.CONV_ARCH:
            self._init_placeholder()

        # Set the operations to reset the optimier when needed
        self._reset_optimizer_ops()

        self._get_graph()
        if self.pretrained:
            self._assign_pretrained_ops()

        # test if the model is setup properly
        self._is_properly_setup()

    #################################################################################
    #### External APIs of the class. These will be called/ exposed externally #######
    #################################################################################

    def train_model(self, trainset, valset, sess, no_epochs, batch_size, task_labels=None):
        self.no_epochs = no_epochs
        self.batch_size = batch_size
        self.task_labels = task_labels
        # Build additional computational graph
        self._train_step()
        self._define_acc()

        # init sess
        sess.run(tf.global_variables_initializer())

        # Set the star values to the initial weights, so that we can calculate
        sess.run(self.set_star_vars)
        if self.pretrained:
            # below code imports pretrained parameters to the model (resnet18)
            sess.run(self.load_pretrained_ops)

        train_accs, train_end_task_accs, test_accs, end_task_accs, end_epoch_accs = self.test_init_same_task_size(
            trainset, batch_size, no_epochs)

        #self._train_step()
        for task_id in range(self.n_tasks):

            task_size = trainset[task_id].images.shape[0]
            total_batch = int(np.ceil(task_size * 1.0 / batch_size))
            print('Received {} images, {} labels at task {}'.format(task_size, trainset[task_id].labels.shape[0],
                                                                    task_id))
            print('Unique labels in the task: {}'.format(np.unique(np.nonzero(trainset[task_id].labels)[1])))
            print('Total batch: {}'.format(total_batch))
            # Mask for softmax
            if self.scheme == 'incre' or self.head == 'multi':
                logit_mask = np.zeros(self.out_dim)
                logit_mask[task_labels[task_id]] = 1.0

            # If not the first task then restore weights from previous task
            if task_id > 0:
                #Restore the weights from the star variables
                sess.run(self.restore_weights)

            for epoch in range(no_epochs):
                perm_inds = list(range(task_size))
                np.random.shuffle(perm_inds)
                cur_x_train = trainset[task_id].images[perm_inds]
                cur_y_train = trainset[task_id].labels[perm_inds]

                for i in range(total_batch):
                    start_ind = i * batch_size
                    end_ind = np.min([(i + 1) * batch_size, task_size])
                    batch_x = cur_x_train[start_ind:end_ind, :]
                    batch_y = cur_y_train[start_ind:end_ind, :]
                    if self.scheme == 'incre' or self.head == 'multi':
                        logit_mask[:] = 0
                        logit_mask[task_labels[task_id]] = 1.0
                    if self.args.arch in global_vars.CONV_ARCH:
                        feed_dict = {self.x: batch_x, self.y_: batch_y, self.keep_prob: 0.5,
                                     self.train_phase: True}
                        if self.scheme == 'incre' or self.head == 'multi':
                            feed_dict[self.output_mask] = logit_mask
                        # If first iteration of the first task then set the initial value of the running fisher
                        if task_id == 0 and i == 0 and epoch == 0:
                            sess.run([self.set_initial_running_fisher], feed_dict=feed_dict)
                        # Update fisher after every few iterations
                        if (epoch*total_batch + i + 1) % self.fisher_update_after == 0:
                            sess.run(self.set_running_fisher)
                            sess.run(self.reset_tmp_fisher)

                        _, _, loss = sess.run([self.set_tmp_fisher, self.train, self.reg_loss], feed_dict=feed_dict)
                    else:
                            g_and_v, _, loss = sess.run([self.grads_and_vars, self.train_step, self.cross_entropy],
                                                       feed_dict={self.x: batch_x, self.y_: batch_y})

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

            # calculate fisher and store optimal params
            if (task_id != self.n_tasks - 1) and (self.arch not in global_vars.CONV_ARCH):
                stime = time.time()
                self._compute_fisher(valset[task_id].images)
                etime = time.time()
                print("Time taken to compute Fisher: %.5f" % (etime - stime))
                self._star()

            # Compute the inter-task updates, Fisher/ importance scores etc
            # Don't calculate the task updates for the last task
            if task_id < (self.n_tasks - 1):
                self.task_updates(sess)
                print('\t\t\t\tTask updates after Task%d done!' % (task_id))


            print('done task {}'.format(task_id))
            print(loss)
            utils.acc_store(self, valset, task_id, sess, end_task_accs, None)

        avg_acc_test = avg_acc(end_task_accs)
        # avg_acc_train = avg_acc(train_end_task_accs)
        # print("average train accuracy")
        # print(avg_acc_train)
        print("average test accuracy")
        print(avg_acc_test)
        return end_task_accs, avg_acc_test


    ####################################################################################
    #### Internal APIs of the class. These should not be called/ exposed externally ####
    ####################################################################################
    ####################################################################################
    #### Internal APIs of the class. These should not be called/ exposed externally ####
    ####################################################################################
    def test_init_same_task_size(self, trainset, batch_size, no_epochs):
        test_accs = []
        end_task_accs = []
        end_epoch_accs = []

        #train_grad = np.zeros(result_size)
        # each task has one ndarray with size total_batch * no_epochs
        task_size = max([trainset[i].images.shape[0] for i in range(self.n_tasks)])
        # Total batch for one epoch
        total_batch = int(np.ceil(task_size * 1.0 / batch_size))
        for task in range(self.n_tasks):
            result_size = (self.n_tasks, int(np.ceil(total_batch * no_epochs / self.disp_freq_train)))
            test_accs.append(np.zeros(result_size))
            end_task_accs.append(np.zeros(self.n_tasks))
            end_epoch_accs.append(np.zeros((self.n_tasks, self.no_epochs)))
        train_accs = deepcopy(test_accs)
        train_end_task_accs = deepcopy(end_task_accs)
        return train_accs, train_end_task_accs, test_accs, end_task_accs, end_epoch_accs

    def _init_placeholder(self):
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        self.output_mask = tf.placeholder(dtype=tf.float32, shape=[self.out_dim])
        self.sample_weights = tf.placeholder(tf.float32, shape=[None])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=())

    def _is_properly_setup(self):
        super(EWC, self)._is_properly_setup()
        if self.arch == 'fully' and len(self.in_dim) != 1:
            raise Exception("input size should have length 1 for fully connected ")
        elif 'resnet' in self.arch and len(self.in_dim) != 3:
            raise Exception("input size should have length 3 for CNN ")

    def _get_graph(self):
        if self.arch == 'fully':
            fully_model(self)
        elif 'resnet18' in self.arch:
            kernels, filters, strides = resnet_params(self.arch)
            act = self.x
            resnet18(self, act, kernels, filters, strides)

    def _train_step(self):
        if self.arch == 'fully':
            with tf.variable_scope('vanilla_loss'):
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))
            # elastic weight consolidation
            # lam is weighting for previous task(s) constraints
            name = self.optimizer.get_name()
            if not hasattr(self, "ewc_loss"):
                self.ewc_loss = self.loss
            else:
                for v in range(len(self.trainable_vars)):
                    self.ewc_loss += (self.lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))
            self.grads_and_vars = self.optimizer.compute_gradients(self.ewc_loss, self.trainable_vars)
            vars_with_grad = [v for g, v in self.grads_and_vars if g is not None]
            if not vars_with_grad:
                raise ValueError(
                    "No gradients provided for any variable, check your graph for ops"
                    " that do not support gradients, between variables %s and loss %s." %
                    ([str(v) for _, v in self.grads_and_vars], self.loss))
            self.train_step = self.optimizer.apply_gradients(self.grads_and_vars, name=name)
        else:

            if self.scheme == 'full':
                with tf.variable_scope('vanilla_loss'):
                    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))

            elif self.scheme == 'incre' or self.head == 'multi':
                self.pruned_logits = tf.where(
                    tf.tile(tf.equal(self.output_mask[None, :], 1.0), [tf.shape(self.y)[0], 1]), self.y,
                    global_vars.NEG_INF * tf.ones_like(self.y))

                with tf.variable_scope('vanilla_loss'):
                    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                                                                                    logits=self.pruned_logits))

            self.init_vars()

            reg = tf.add_n([tf.reduce_sum(tf.square(w - w_star) * f) for w, w_star, f in
                            zip(self.trainable_vars, self.star_vars, self.normalized_fisher_at_minima_vars)])

            # Regularized training loss
            with tf.variable_scope('regulated_loss'):
                self.reg_loss = tf.squeeze(self.loss + self.lamda * reg)
            # Compute the gradients of the vanilla loss
            self.vanilla_gradients_vars = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_vars)
            # Compute the gradients of regularized loss
            self.reg_gradients_vars = self.optimizer.compute_gradients(self.reg_loss, var_list=self.trainable_vars)


            # Store the current weights before doing a train step
            self.get_current_weights()

            # Get the value of old weights first
            with tf.control_dependencies([self.weights_old_ops_grouped]):
                # Define a training operation
                self.train = self.optimizer.apply_gradients(self.reg_gradients_vars)

            # Create operations to compute importance
            self.create_fisher_ops()

            # Create weight save and store ops
            self.weights_store_ops()

            # Summary operations for visualization
            # tf.summary.scalar("loss", self.loss)
            # for v in self.trainable_vars:
            #     tf.summary.histogram(v.name.replace(":", "_"), v)
            # self.merged_summary = tf.summary.merge_all()

    def _reset_optimizer_ops(self):
        """
        Defines operations to reset the optimizer
        Args:

        Returns:
        """
        # Set the operation for resetting the optimizer
        self.optimizer_slots = [self.optimizer.get_slot(var, name) for name in self.optimizer.get_slot_names()\
                           for var in tf.global_variables() if self.optimizer.get_slot(var, name) is not None]
        self.slot_names = self.optimizer.get_slot_names()
        self.opt_init_op = tf.variables_initializer(self.optimizer_slots)

    def _define_acc(self):
        if self.args.head == 'single' and self.scheme == 'full':
            with tf.variable_scope('accuracy'):
                # single head setting
                self.correct_predictions = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))
        else:
            with tf.variable_scope('accuracy'):
                # multi head setting
                self.correct_predictions = tf.equal(tf.argmax(self.pruned_logits, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))


    def grad_mean(self, grads_and_vars):
        return self.grad_abs_sum(grads_and_vars) / self.num_param

    @staticmethod
    def grad_abs_sum(grads_and_vars):
        sum_grads = 0
        for (g, v) in grads_and_vars:
            sum_grads += np.sum(np.absolute(g))
        return sum_grads

    @property
    def model_name(self):
        if self.args.data == 'mnist':
            return "EWC_data{}_arch{}_embed{}_epoch{}_batch{}_optmz{}_lr{}_dis{}_numtasks{}_pretrain{}_scheme{}". \
                format(self.args.data, self.arch, self.embed_dim, self.no_epochs, self.batch_size, self.args.optimizer,
                       self.learning_rate,  self.disp_freq_train, self.n_tasks,
                       self.pretrained, self.scheme)
        else:
            return "EWC_data{}_arch{}_epoch{}_batch{}_optmz{}_lr{}_dis{}_numtasks{}_pretrain{}_scheme{}_lama{}". \
                format(self.args.data, self.arch, self.no_epochs, self.batch_size, self.args.optimizer,
                       self.learning_rate,  self.disp_freq_train, self.n_tasks,
                       self.pretrained, self.scheme, self.args.lambda_)

    @property
    def name(self):
        return "EWC"

    def _get_num_param(self):
        assert(hasattr(self, 'var_list')), "self.var_list not defined"
        num = 0
        for v in self.var_list:
            num_v = 1
            shape = v.get_shape()
            for dim in shape:
                num_v *= dim.value
            num += num_v
        return num

    def init_vars(self):
        """
        Defines different variables that will be used for the
        weight consolidation
        Args:

        Returns:
        """
        # Define different variables
        self.weights_old = []
        self.star_vars = []

        self.fisher_diagonal_at_minima = []

        self.running_fisher_vars = []
        self.tmp_fisher_vars = []
        self.max_fisher_vars = []
        self.min_fisher_vars = []
        self.max_score_vars = []
        self.min_score_vars = []
        self.normalized_score_vars = []
        self.score_vars = []
        self.normalized_fisher_at_minima_vars = []
        self.weights_delta_old_vars = []
        self.ref_grads = []
        self.projected_gradients_list = []

        for v in range(len(self.trainable_vars)):

            # List of variables for weight updates
            self.weights_old.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.weights_delta_old_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.star_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False, name=self.trainable_vars[v].name.rsplit(':')[0]+'_star'))


            # List of variables to store fisher information
            self.fisher_diagonal_at_minima.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

            self.normalized_fisher_at_minima_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False, dtype=tf.float32))
            self.tmp_fisher_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.running_fisher_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            # New variables for conv setting for fisher and score normalization
            self.max_fisher_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.min_fisher_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.max_score_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.min_score_vars.append(tf.Variable(tf.zeros(1), dtype=tf.float32, trainable=False))
            self.normalized_score_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

    def get_current_weights(self):
        """
        Get the values of current weights
        Note: These weights are different from star_vars as those
        store the weights after training for the last task.
        Args:

        Returns:
        """
        weights_old_ops = []
        weights_delta_old_ops = []
        for v in range(len(self.trainable_vars)):
            weights_old_ops.append(tf.assign(self.weights_old[v], self.trainable_vars[v]))
            weights_delta_old_ops.append(tf.assign(self.weights_delta_old_vars[v], self.trainable_vars[v]))

        self.weights_old_ops_grouped = tf.group(*weights_old_ops)
        self.weights_delta_old_grouped = tf.group(*weights_delta_old_ops)


    def create_fisher_ops(self):
        """
        Defines the operations to compute online update of Fisher
        Args:

        Returns:
        """
        ders = tf.gradients(self.loss, self.trainable_vars)
        fisher_ema_at_step_ops = []
        fisher_accumulate_at_step_ops = []

        # ops for running fisher
        self.set_tmp_fisher = [tf.assign_add(f, tf.square(d)) for f, d in zip(self.tmp_fisher_vars, ders)]

        # Initialize the running fisher to non-zero value
        self.set_initial_running_fisher = [tf.assign(r_f, s_f) for r_f, s_f in zip(self.running_fisher_vars,
                                                                           self.tmp_fisher_vars)]

        self.set_running_fisher = [tf.assign(f, (1 - self.fisher_ema_decay) * f + (1.0/ self.fisher_update_after) *
                                    self.fisher_ema_decay * tmp) for f, tmp in zip(self.running_fisher_vars, self.tmp_fisher_vars)]

        self.get_fisher_at_minima = [tf.assign(var, f) for var, f in zip(self.fisher_diagonal_at_minima,
                                                                         self.running_fisher_vars)]

        self.reset_tmp_fisher = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.tmp_fisher_vars]

        # Get the min and max in each layer of the Fisher
        self.get_max_fisher_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_max(scr, keepdims=True)), axis=0))
                for var, scr in zip(self.max_fisher_vars, self.fisher_diagonal_at_minima)]
        self.get_min_fisher_vars = [tf.assign(var, tf.expand_dims(tf.squeeze(tf.reduce_min(scr, keepdims=True)), axis=0))
                for var, scr in zip(self.min_fisher_vars, self.fisher_diagonal_at_minima)]
        self.max_fisher = tf.reduce_max(tf.convert_to_tensor(self.max_fisher_vars))
        self.min_fisher = tf.reduce_min(tf.convert_to_tensor(self.min_fisher_vars))
        with tf.control_dependencies([self.max_fisher, self.min_fisher]):
            self.normalize_fisher_at_minima = [tf.assign(tgt,
                (var - self.min_fisher)/ (self.max_fisher - self.min_fisher + global_vars.EPSILON))
                    for tgt, var in zip(self.normalized_fisher_at_minima_vars, self.fisher_diagonal_at_minima)]

        self.clear_attr_embed_reg = tf.assign(self.normalized_fisher_at_minima_vars[-2], tf.zeros_like(self.normalized_fisher_at_minima_vars[-2]))

        # Sparsify all the layers except last layer
        sparsify_fisher_ops = []
        for v in range(len(self.normalized_fisher_at_minima_vars) - 2):
            sparsify_fisher_ops.append(tf.assign(self.normalized_fisher_at_minima_vars[v],
                tf.nn.dropout(self.normalized_fisher_at_minima_vars[v], self.keep_prob)))

        self.sparsify_fisher = tf.group(*sparsify_fisher_ops)


    def weights_store_ops(self):
        """
        Defines weight restoration operations
        Args:

        Returns:
        """
        restore_weights_ops = []
        set_star_vars_ops = []

        for v in range(len(self.trainable_vars)):
            restore_weights_ops.append(tf.assign(self.trainable_vars[v], self.star_vars[v]))

            set_star_vars_ops.append(tf.assign(self.star_vars[v], self.trainable_vars[v]))

        self.restore_weights = tf.group(*restore_weights_ops)
        self.set_star_vars = tf.group(*set_star_vars_ops)

    def task_updates(self, sess):
        """
        Updates different variables when a task is completed
        Args:
            sess                TF session
            task                Task ID
            train_x             Training images for the task
            train_labels        Labels in the task
            class_attr          Class attributes (only needed for ZST transfer)
        Returns:
        """
        # Get the fisher at the end of a task
        sess.run(self.get_fisher_at_minima)
        # Normalize the fisher
        sess.run([self.get_max_fisher_vars, self.get_min_fisher_vars])
        sess.run([self.min_fisher, self.max_fisher, self.normalize_fisher_at_minima])
        # Reset the tmp fisher vars
        sess.run(self.reset_tmp_fisher)
        # Store current weights
        sess.run(self.set_star_vars)

    ############for MNIST, need to update
    def _compute_fisher(self, imgset):
        pass
        # # computer Fisher information for each parameter
        #
        # # initialize Fisher information for most recent task
        # print("calculating fisher")
        # self.F_accum = []
        # for v in range(len(self.var_list)):
        #     self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
        #
        # # sampling a random class from softmax
        # probs = tf.nn.softmax(self.y)
        # class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])
        # fish_gra = tf.gradients(tf.log(probs[0, class_ind]), self.var_list)
        # for i in range(self.fisher_num_samples):
        #     # select random input image
        #     im_ind = np.random.randint(imgset.shape[0])
        #     # compute first-order derivatives
        #     ders = self.sess.run(fish_gra, feed_dict={self.x: imgset[im_ind:im_ind + 1]})
        #
        #     # square the derivatives and add to total
        #     for v in range(len(self.F_accum)):
        #         self.F_accum[v] += np.square(ders[v])
        #
        # # divide totals by number of samples
        # for v in range(len(self.F_accum)):
        #     self.F_accum[v] /= self.fisher_num_samples


    def _star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval(session=self.sess))


    def _restore(self):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                self.sess.run(self.var_list[v].assign(self.star_vars[v]))


    def _set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cross_entropy)

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
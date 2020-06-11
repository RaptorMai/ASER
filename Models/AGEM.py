import tensorflow as tf
import numpy as np
from copy import deepcopy
import os, sys
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
import utils.utils as utils
from Models.model_wrapper import CLInterface
from Metrics.metrics import avg_acc
from utils.build_architecture import fully_model, resnet18, resnet_params
import utils.global_vars as global_vars
from utils.resnet_utils import load_resnet18

class AGEM(CLInterface):
    def __init__(self,
                 input_size,
                 output_size,
                 learning_rate=0.03,
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
        super(AGEM, self).__init__()
        if arch not in global_vars.VALID_ARCH:
            raise Exception('not valid architecture')
        else:
            self.arch = arch
        self.pretrained = extra_arg.pretrained
        self.in_dim = input_size
        self.out_dim = output_size
        if self.pretrained:
            self.x = tf.placeholder(tf.float32, shape=[None]+[224,224,3])
        else:
            self.x = tf.placeholder(tf.float32, shape=[None]+ input_size)
        self.y_ = tf.placeholder(tf.float32, shape=[None, output_size])
        self.train_phase = tf.placeholder(tf.bool, name='train_phase')
        self.learning_rate = learning_rate
        self.disp_freq_train = disp_freq_train
        self.optimizer = optimizer(learning_rate)
        self.head = head
        self.scheme = scheme
        self.n_tasks = extra_arg.num_tasks
        self.args = extra_arg

        #according to author of AGEM, this is true only when we use CUB dataset
        self.is_ATT_DATASET = is_ATT_DATASET

        # Class attributes for zero shot transfer
        self.class_attr = attr
        self.args = extra_arg


        self._init_placeholder()

        #mem related
        self.mem_size = extra_arg.mem_size
        self.eps_mem_batch = extra_arg.eps_mem_batch

        # test if the model is setup properly
        self._is_properly_setup()

        # additional setup
        self.violation_count = tf.Variable(0, dtype=tf.float32, trainable=False)

        # setup the graph
        self._get_graph()

        # Define variables necessary for AGEM algorithm (if turn out to be unnecessary, remove them)
        self.weights_old = []
        self.star_vars = []
        self.weights_delta_old_vars = []

        # Build additional computational graph
        self._train_step()
        self._define_acc()

        self.num_param = self._get_num_param()  # ignored when pretrained model is used
        if self.pretrained:
            self._assign_pretrained_ops()

        # Set the operations to reset the optimizer when needed
        self._reset_optimizer_ops()


    #################################################################################
    #### External APIs of the class. These will be called/ exposed externally #######
    #################################################################################

    def train_model(self, trainset, valset, sess, no_epochs, batch_size, task_labels=None):

        self.no_epochs = no_epochs
        self.batch_size = batch_size
        self._define_mem(self.n_tasks)
        self.task_labels = task_labels
        logit_mask = np.ones(self.out_dim)          # Might change depending on single-head v.s. multi-head

        train_accs, train_end_task_accs, test_accs, end_task_accs = \
            self.test_init_same_task_size(trainset, batch_size, no_epochs)

        if self.args.data in global_vars.SPLIT_DATA:
            # List to store the classes that we have so far - used at test time
            test_labels = []

        # init sess
        sess.run(tf.global_variables_initializer())

        # Run the init ops
        self.init_updates(sess)

        if self.pretrained:
            # below code imports pretrained parameters to the model (resnet18)
            sess.run(self.load_pretrained_ops)

        for task_id in range(self.n_tasks):
            if self.args.data in global_vars.SPLIT_DATA:
                # Test for the tasks that we've seen so far
                test_labels += task_labels[task_id]

            for epoch in range(no_epochs):
                task_size = trainset[task_id].images.shape[0]
                total_batch = int(np.ceil(task_size * 1.0 / batch_size))
                print('Received {} images, {} labels at task {}'.format(task_size, trainset[task_id].labels.shape[0], task_id))
                print('Unique labels in the task: {}'.format(np.unique(np.nonzero(trainset[task_id].labels)[1])))

                perm_inds = list(range(task_size))
                np.random.shuffle(perm_inds)
                cur_x_train = trainset[task_id].images[perm_inds]
                cur_y_train = trainset[task_id].labels[perm_inds]

                for i in range(total_batch):
                    start_ind = i * batch_size
                    end_ind = np.min([(i+1) * batch_size, task_size])
                    batch_x = cur_x_train[start_ind:end_ind, :]
                    batch_y = cur_y_train[start_ind:end_ind, :]

                    feed_dict = {self.x: batch_x, self.y_: batch_y, self.train_phase: True}
                    if self.scheme in ['indep', 'incre']:
                        feed_dict.update({m_t: i_t for (m_t, i_t) in zip(self.output_mask, logit_mask)})

                    if task_id == 0:
                        if self.scheme in ['indep', 'incre'] and self.args.data in global_vars.SPLIT_DATA:
                            self.nd_logit_mask[:] = 0
                            self.nd_logit_mask[task_id][task_labels[task_id]] = 1.0
                            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(self.output_mask, self.nd_logit_mask)}
                            feed_dict.update(logit_mask_dict)
                            feed_dict[self.mem_batch_size] = batch_size

                        # Normal application of gradients
                        _, loss = sess.run([self.train_first_task, self.agem_loss],
                                                feed_dict=feed_dict)
                    else:
                        # Compute and store the reference gradients on the previous tasks
                        if self.args.data in global_vars.SPLIT_DATA:
                            # Set the mask for all the previous tasks so far
                            self.nd_logit_mask[:] = 0
                            for tt in range(task_id):
                                self.nd_logit_mask[tt][task_labels[tt]] = 1.0

                        if self.episodic_filled_counter <= self.eps_mem_batch:
                            mem_sample_mask = np.arange(self.episodic_filled_counter)
                        else:
                            # Sample a random subset from episodic memory buffer
                            mem_sample_mask = np.random.choice(self.episodic_filled_counter, self.eps_mem_batch,
                                                               replace=False)  # Sample without replacement so that we don't sample an example more than once
                        ref_feed_dict = {self.x: self.episodic_images[mem_sample_mask],
                                     self.y_: self.episodic_labels[mem_sample_mask], self.train_phase: True}

                        if self.scheme in ['indep', 'incre'] and self.args.data in global_vars.SPLIT_DATA:
                            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(self.output_mask, self.nd_logit_mask)}
                            ref_feed_dict.update(logit_mask_dict)
                            ref_feed_dict[self.mem_batch_size] = float(len(mem_sample_mask))
                        # Store the reference gradient
                        sess.run(self.store_ref_grads, feed_dict=ref_feed_dict)

                        # Compute the gradient for current task and project if need be
                        if self.scheme in ['indep', 'incre'] and self.args.data in global_vars.SPLIT_DATA:
                            if self.scheme == 'indep':
                                self.nd_logit_mask[:] = 0
                                self.nd_logit_mask[task_id][task_labels[task_id]] = 1.0
                            elif self.scheme == 'incre':
                                self.nd_logit_mask[:] = 0
                                for tt in range(task_id):
                                    self.nd_logit_mask[tt][task_labels[tt]] = 1.0
                            logit_mask_dict = {m_t: i_t for (m_t, i_t) in zip(self.output_mask, self.nd_logit_mask)}
                            feed_dict.update(logit_mask_dict)
                            feed_dict[self.mem_batch_size] = batch_size
                        _, loss = sess.run([self.train_subseq_tasks, self.agem_loss], feed_dict=feed_dict)

                    # Put the batch in the ring buffer
                    for er_x, er_y_ in zip(batch_x, batch_y):
                        cls = np.unique(np.nonzero(er_y_))[-1]
                        # Write the example at the location pointed by count_cls[cls]
                        if self.args.data in global_vars.SPLIT_DATA:
                            cls_to_index_map = np.where(np.array(task_labels[task_id]) == cls)[0][0]
                        else:
                            cls_to_index_map = cls
                        with_in_task_offset = self.mem_size * cls_to_index_map
                        mem_index = self.count_cls[cls] + with_in_task_offset + self.episodic_filled_counter
                        self.episodic_images[mem_index] = er_x
                        self.episodic_labels[mem_index] = er_y_
                        self.count_cls[cls] = (self.count_cls[cls] + 1) % self.mem_size


                    # test on valset
                    if (epoch * total_batch + i) % self.disp_freq_train == 0:
                        # Compute the mean gradient of all trainable parameters
                        print('Step {}: loss {}'.format((epoch * total_batch + i), loss))

                        # Compute the mean gradient of all trainable parameters
                        #train_grad[task_id, int((epoch * total_batch + i) / self.disp_freq_train)] = self.grad_mean(g_and_v)
                        index = int((epoch * total_batch + i) / self.disp_freq_train)
                        utils.acc_store(self, valset, task_id, sess, test_accs, index)
            print('done task {}'.format(task_id))
            print(loss)
            self.init_updates(sess)
            utils.acc_store(self, valset, task_id, sess, end_task_accs, None)

            if self.args.data in global_vars.SPLIT_DATA:
                self.episodic_filled_counter += self.mem_size * len(task_labels[0])
            else:
                self.episodic_filled_counter += self.mem_size * self.out_dim

            print('done task {}'.format(task_id))
        avg_acc_test = avg_acc(end_task_accs)
        print("average accuracy")
        print(avg_acc_test)
        # utils.store_results("AGEM", self.model_name, test_acc=test_accs)
        # #utils.store_results("AGEM", self.model_name, train_grad=train_grad)
        # utils.store_results("AGEM", self.model_name, end_task_accs=end_task_accs)
        # utils.store_results("AGEM", self.model_name, avg_acc_=avg_acc_test)
        return end_task_accs, avg_acc_test




    ####################################################################################
    #### Internal APIs of the class. These should not be called/ exposed externally ####
    ####################################################################################
    def test_init_same_task_size(self, trainset, batch_size, no_epochs):
        test_accs = []
        end_task_accs = []
        #TODO
        #train_grad = np.zeros(result_size)
        # each task has one ndarray with size total_batch * no_epochs
        task_size = max([trainset[i].images.shape[0] for i in range(self.n_tasks)])
        # Total batch for one epoch
        total_batch = int(np.ceil(task_size * 1.0 / batch_size))
        for task in range(self.n_tasks):
            result_size = (self.n_tasks, int(np.ceil(total_batch * no_epochs / self.disp_freq_train)))
            test_accs.append(np.zeros(result_size))
            end_task_accs.append(np.zeros(self.n_tasks))
        train_accs = deepcopy(test_accs)
        train_end_task_accs = deepcopy(end_task_accs)
        return train_accs, train_end_task_accs, test_accs, end_task_accs

    def init_updates(self, sess):
        sess.run(self.set_star_vars)

    def _get_graph(self):

        if self.arch == 'fully':
            fully_model(self)
        elif 'resnet18' in self.arch:
            kernels, filters, strides = resnet_params(self.arch)
            act = self.x
            resnet18(self, act, kernels, filters, strides)

    def _define_mem(self, num_task):

        if self.args.data not in global_vars.SPLIT_DATA:
            self.episodic_mem_size = self.mem_size * self.out_dim * num_task
        else:
            self.episodic_mem_size = self.mem_size * self.out_dim

        self.episodic_images = np.zeros([self.episodic_mem_size]+ self.in_dim)
        self.episodic_labels = np.zeros([self.episodic_mem_size, self.out_dim])
        self.nd_logit_mask = np.zeros([num_task, self.out_dim])
        self.count_cls = np.zeros(self.out_dim, dtype=np.int32)
        self.episodic_filled_counter = 0
        self.examples_seen_so_far = 0

    def _init_var_gradients(self):
        self.ref_grads = []
        self.projected_gradients_list = []
        for v in range(len(self.trainable_vars)):
            # List of variables for weight updates
            self.weights_old.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.weights_delta_old_vars.append(
                tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.star_vars.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False,
                                              name=self.trainable_vars[v].name.rsplit(':')[0] + '_star'))
            self.ref_grads.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))
            self.projected_gradients_list.append(tf.Variable(tf.zeros(self.trainable_vars[v].get_shape()), trainable=False))

    def _get_current_weights(self):
        """
            Get the values of current weights
            Note: These weights are different from star_vars as those
            store the weights after training for the last task.
            Args:
        """
        weights_old_ops = []
        weights_delta_old_ops = []
        for v in range(len(self.trainable_vars)):
            weights_old_ops.append(tf.assign(self.weights_old[v], self.trainable_vars[v]))
            weights_delta_old_ops.append(tf.assign(self.weights_delta_old_vars[v], self.trainable_vars[v]))

        self.weights_old_ops_grouped = tf.group(*weights_old_ops)
        self.weights_delta_old_grouped = tf.group(*weights_delta_old_ops)

    def _train_step(self):
        """
        Define operations for A GEM
        """
        if self.arch == 'fully':
            if self.head == 'single':
                self.logits = self.y
                self._init_var_gradients()
                self.mse = 2.0 * tf.nn.l2_loss(self.logits)
                self.unweighted_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.logits))
            else:
                self.output_mask = tf.placeholder(dtype=tf.float32, shape=[self.out_dim])
                self.logits = tf.where(tf.tile(tf.equal(self.output_mask[None, :], 1.0), [tf.shape(self.y)[0], 1]), self.y, global_vars.NEG_INF*tf.ones_like(self.y))
                self._init_var_gradients()
                self.mse = 2.0 * tf.nn.l2_loss(self.pruned_logits)  # tf.nn.l2_loss computes sum(T**2)/ 2
                self.unweighted_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.pruned_logits))

        elif 'resnet' in self.arch:
            if self.scheme == 'full':
                self.logits = self.y

            else:
                # output mask only for split tasks
                self.output_mask = [tf.placeholder(dtype=tf.float32, shape=[self.out_dim]) for i in range(self.n_tasks)]
                self.task_pruned_logits = []
                self.unweighted_entropy = []
                for i in range(self.n_tasks):
                    self.task_pruned_logits.append(
                        tf.where(tf.tile(tf.equal(self.output_mask[i][None, :], 1.0), [tf.shape(self.y)[0], 1]), self.y,
                                 global_vars.NEG_INF * tf.ones_like(self.y)))
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_,
                                                                               logits=self.task_pruned_logits[i])
                    adjusted_entropy = tf.reduce_sum(
                        tf.cast(tf.tile(tf.equal(self.output_mask[i][None, :], 1.0), [tf.shape(self.y_)[0], 1]),
                                dtype=tf.float32) * self.y_, axis=1) * cross_entropy
                    self.unweighted_entropy.append(tf.reduce_sum(adjusted_entropy))

            self._init_var_gradients()

        self._get_current_weights()  # Store the current weights before doing a train step

        if self.arch == 'fully':
            self.agem_loss = self.unweighted_entropy
        elif 'resnet' in self.arch:
            if self.scheme == 'full':
                with tf.variable_scope('vanila_loss'):
                    self.agem_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.logits))
            else:
                with tf.variable_scope('vanilla_loss'):
                    self.agem_loss = tf.add_n(
                        [self.unweighted_entropy[i] for i in range(self.n_tasks)]) / self.mem_batch_size


        #fully code before
        ref_grads = tf.gradients(self.agem_loss, self.trainable_vars)

        # Reference gradient for previous tasks
        self.store_ref_grads = [tf.assign(ref, grad) for ref, grad in zip(self.ref_grads, ref_grads)]
        flat_ref_grads = tf.concat([tf.reshape(grad, [-1]) for grad in self.ref_grads], 0)  # vectorize all ref gradients

        # Grandient on the current task
        task_grads = tf.gradients(self.agem_loss, self.trainable_vars)
        flat_task_grads = tf.concat([tf.reshape(grad, [-1]) for grad in task_grads], 0)     # vectorize all task gradients

        # below block ensures flat_task_grads to be computed prior to any codes within the block
        with tf.control_dependencies([flat_task_grads]):
            dotp = tf.reduce_sum(tf.multiply(flat_task_grads, flat_ref_grads))              # dot product: g_ref dot g
            ref_mag = tf.reduce_sum(tf.multiply(flat_ref_grads, flat_ref_grads))            # normalizer
            proj = flat_task_grads - ((dotp/ ref_mag) * flat_ref_grads)                     # projected gradient
            self.reset_violation_count = self.violation_count.assign(0)
            def increment_violation_count():
                with tf.control_dependencies([tf.assign_add(self.violation_count, 1)]):
                    return tf.identity(self.violation_count)

            # project gradients only when they are negative
            self.violation_count = tf.cond(tf.greater_equal(dotp, 0), lambda: tf.identity(self.violation_count), increment_violation_count)
            projected_gradients = tf.cond(tf.greater_equal(dotp, 0), lambda: tf.identity(flat_task_grads), lambda: tf.identity(proj))

            # Convert the flat projected gradient vector into a list
            offset = 0
            store_proj_grad_ops = []
            for v in self.projected_gradients_list:
                shape = v.get_shape()
                v_params = 1
                for dim in shape:
                    v_params *= dim.value
                store_proj_grad_ops.append(tf.assign(v, tf.reshape(projected_gradients[offset:offset+v_params], shape)))
                offset += v_params
            self.store_proj_grads = tf.group(*store_proj_grad_ops)

            # Define training operations for the tasks > 1
            # Due to the dependency structure, tf.store_proj_grads is evaluated (which assigns all projected gradients
            # corresponding to variables in self.projected_gradients_list),
            # and then we apply the gradients to the variables
            with tf.control_dependencies([self.store_proj_grads]):
                self.train_subseq_tasks = self.optimizer.apply_gradients(zip(self.projected_gradients_list, self.trainable_vars))

        # Define training operations for the first task
        self.first_task_gradients_vars = self.optimizer.compute_gradients(self.agem_loss, var_list=self.trainable_vars)
        self.train_first_task = self.optimizer.apply_gradients(self.first_task_gradients_vars)

        # Create weight save and store ops
        self.weights_store_ops()

        # Summary operations for visualization
        # tf.summary.scalar("unweighted_entropy", self.unweighted_entropy)
        # for v in self.trainable_vars:
        #     tf.summary.histogram(v.name.replace(":", "_"), v)
        # self.merged_summary = tf.summary.merge_all()


    def weights_store_ops(self):
        restore_weights_ops = []
        set_star_vars_ops = []

        for v in range(len(self.trainable_vars)):
            restore_weights_ops.append(tf.assign(self.trainable_vars[v], self.star_vars[v]))
            set_star_vars_ops.append(tf.assign(self.star_vars[v], self.trainable_vars[v]))
        self.restore_weights = tf.group(*restore_weights_ops)
        self.set_star_vars = tf.group(*set_star_vars_ops)

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
            return "AGEM_data{}_arch{}_embed{}_epoch{}_batch{}_optmz{}_lr{}_memSize{}_memSizeBatch{}_dis{}_{}_numtasks{}_scheme{}".\
                format(self.args.data, self.arch, self.embed_dim, self.no_epochs, self.batch_size, self.args.optimizer,
                       self.learning_rate, self.mem_size, self.eps_mem_batch, self.disp_freq_train, self.head, self.n_tasks, self.scheme)
        else:
            return "AGEM_data{}_arch{}_epoch{}_batch{}_optmz{}_lr{}_memSize{}_memSizeBatch{}_dis{}_{}_numtasks{}_scheme{}". \
                format(self.args.data, self.arch, self.no_epochs, self.batch_size, self.args.optimizer,
                       self.learning_rate, self.mem_size, self.eps_mem_batch, self.disp_freq_train, self.head,  self.n_tasks, self.scheme)
    def _get_num_param(self):
        assert(hasattr(self, 'trainable_vars')), "self.var_list not defined"
        num = 0
        for v in self.trainable_vars:
            num_v = 1
            shape = v.get_shape()
            for dim in shape:
                num_v *= dim.value
            num += num_v
        return num

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


    def _init_placeholder(self):
        # current mem+batch size
        self.mem_batch_size = tf.placeholder(dtype=tf.float32, shape=())

    def _define_acc(self):
        with tf.variable_scope('accuracy'):
            if self.head == 'single':
                correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            elif self.arch == 'fully':
                # performance metrics
                correct_prediction = tf.equal(tf.argmax(self.pruned_logits, 1), tf.argmax(self.y_, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            else:
                self.correct_predictions = []
                self.accuracy = []
                for i in range(self.n_tasks):
                    self.correct_predictions.append(tf.equal(tf.argmax(self.task_pruned_logits[i], 1), tf.argmax(self.y_, 1)))
                    self.accuracy.append(tf.reduce_mean(tf.cast(self.correct_predictions[i], tf.float32)))

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
import tensorflow as tf
import numpy as np
from Data.data_utils import generate_data
from utils.utils import store_results
from Metrics.metrics import compute_performance
from scipy.stats import sem
import scipy.stats as stats

def multiple_run(final_params, iid=False, save=False):
    end_task_accs_list, avg_acc_test_list, file = [], [], None
    for i in range(final_params.num_runs):
        print("start run {}".format(i))
        # prepare data
        trainset, valset, testset, task_labels = generate_data(final_params, final_params.random_seed + i)

        # Reset the default graph
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            # Set the random seed
            tf.set_random_seed(final_params.random_seed + i)

            # Set up tf session and initialize variables.
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            # model definition
            model = final_params.model_object(
                input_size=final_params.input_size,
                output_size=final_params.output_size,
                learning_rate=final_params.learning_rate,
                disp_freq_train=final_params.disp_freq_train,
                optimizer=final_params.optimizer_object,
                arch=final_params.arch,
                head=final_params.head,
                scheme=final_params.train_scheme,
                extra_arg=final_params
            )

            with tf.Session(config=config, graph=graph) as sess:
                if final_params.combine_val:
                    end_task_accs, avg_acc_test = model.train_model(trainset, testset, sess, final_params.epoch,
                                                                    final_params.batch,
                                                                    task_labels=task_labels)
                else:
                    end_task_accs, avg_acc_test = model.train_model(trainset, valset, sess, final_params.epoch, final_params.batch,
                                                                task_labels=task_labels)

                sess.close()
        end_task_accs_list.append(end_task_accs)
        avg_acc_test_list.append(avg_acc_test)

    end_task_accs_arr = np.array(end_task_accs_list)
    if iid:
        n_run = len(end_task_accs_arr)
        t_coef = stats.t.ppf((1 + 0.95) / 2,
                             n_run - 1)  # t coefficient used to compute 95% CIs: mean +- t * standard error

        acc = np.array(end_task_accs_arr)  # array of shape (num_run, num_task, num_task)

        # compute average test accuracy and CI
        end_acc = acc[:, :, -1]  # shape: (num_run, num_task)
        avg_acc_per_run = np.mean(end_acc, axis=1)  # mean of end task accuracies per run
        avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))
        return end_task_accs_arr, avg_end_acc, None, None, None
    else:
        avg_end_acc, forgetting, avg_acc_task = compute_performance(end_task_accs_arr)
        if save:
            file = store_results(final_params.model, model.model_name, end_task_accs_arr=end_task_accs_arr)
        return end_task_accs_arr, avg_end_acc, forgetting, avg_acc_task, file



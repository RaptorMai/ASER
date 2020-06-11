import numpy as np
from scipy.stats import sem
import scipy.stats as stats

def avg_acc(end_task_acc):
    end_acc = np.array(end_task_acc)
    ret = []
    for i in range(end_acc.shape[0]):
        ret.append(np.mean(end_acc[0:i + 1, i]))
    return ret

def compute_performance(end_task_acc_arr, task_ids=None):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run = len(end_task_acc_arr)
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t * standard error

    acc = np.array(end_task_acc_arr)                # array of shape (num_run, num_task, num_task)

    # compute average test accuracy and CI
    end_acc = acc[:, :, -1]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute average of accuracies on task i (given as task_id)
    task_ids = [task_ids] if isinstance(task_ids, int) else task_ids
    acc_task = acc[:, task_ids, :]                  # shape: (num_run, *, num_task)
    avg_acc_task = (np.mean(acc_task, axis=0), t_coef * sem(acc_task, axis=0))  # shape of each: (*, num_task)

    # compute forgetting
    tmp = acc - acc[:, :, -1].reshape(n_run, -1, 1)         # subtract the end task accuracy from each element
    fgt = tmp[:, :, :-1].max(axis=-1)                       # shape: (num_run, num_task) -- forgetting of each task in each run
    avg_fgt = np.mean(fgt[:, :-1], axis=1)                  # average forgetting in a single run (excl. last task)
    forgetting = (np.mean(avg_fgt), t_coef * sem(avg_fgt))  # average forgetting over multiple runs with CI
    return avg_end_acc, forgetting, avg_acc_task
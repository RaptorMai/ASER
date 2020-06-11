from utils.io import load_yaml
from utils.names_match import models, optimizers, input_sizes, output_sizes
import argparse
import time
from types import SimpleNamespace
from experiment.multiple_run import multiple_run
from utils.io import boolean_string

def main(args):
    defaul_params = load_yaml(args.parameters)
    data_name = defaul_params['data']
    model = defaul_params['model']
    defaul_params['input_size'] = input_sizes[data_name]
    defaul_params['output_size'] = output_sizes[data_name]
    defaul_params['optimizer_object'] = optimizers[defaul_params['optimizer']]
    defaul_params['model_object'] = models[model]
    defaul_params['data'] = data_name
    defaul_params['model'] = model
    args_params = SimpleNamespace(**defaul_params)
    print(args_params)
    time_start = time.time()
    end_task_accs_arr, avg_end_acc, forgetting, avg_acc_task, file = multiple_run(args_params, iid=args.iid, save=True)
    print(avg_end_acc, forgetting)
    time_end = time.time()
    time_spend = time_end - time_start
    print("Time spent {}".format(str(time_spend)))

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="ParameterTuning")

    parser.add_argument('--parameters', dest='parameters', default='config/default.yml')
    parser.add_argument('--iid', dest='iid', type=boolean_string, default=False)
    args = parser.parse_args()

    main(args)

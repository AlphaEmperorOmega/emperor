from models.parser import get_experiment_parser, resolve_experiment_mode
from models.parametric_matrix import Experiment, ExperimentOptions

if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentOptions.names())
    args = parser.parse_args()
    config_option, num_samples = resolve_experiment_mode(args, ExperimentOptions)
    experiment = Experiment(config_option)
    experiment.train_model(num_samples=num_samples, log_folder=args.log_folder)

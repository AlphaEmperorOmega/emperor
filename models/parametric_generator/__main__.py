from models.parser import get_experiment_parser, resolve_experiment_mode
from models.parametric_generator import Experiment, ExperimentOptions

if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentOptions.names())
    args = parser.parse_args()
    config_option, search_mode = resolve_experiment_mode(args, ExperimentOptions)
    experiment = Experiment(config_option)
    experiment.train_model(search_mode=search_mode, log_folder=args.logdir)

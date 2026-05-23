from models.parser import get_experiment_parser, resolve_experiment_mode
from models.experts_linear_adaptive import Experiment, ExperimentOptions

if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentOptions.names(), "models.experts_linear_adaptive")
    args = parser.parse_args()
    config_option, search_mode, search_keys, config_overrides, search_overrides = resolve_experiment_mode(
        args,
        ExperimentOptions,
        no_search_options=[],
    )
    experiment = Experiment(config_option)
    experiment.train_model(
        search_mode=search_mode,
        log_folder=args.logdir,
        search_keys=search_keys,
        config_overrides=config_overrides,
        search_overrides=search_overrides,
    )

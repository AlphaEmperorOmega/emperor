from models.parser import get_experiment_parser, resolve_dataset_names, resolve_experiment_mode
from models.parametric.parametric_vector import Experiment, ExperimentOptions

if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentOptions.names(), "models.parametric.parametric_vector")
    args = parser.parse_args()
    config_option, selected_options, search_mode, search_keys, config_overrides, search_overrides = resolve_experiment_mode(
        args, ExperimentOptions
    )
    experiment = Experiment(config_option)
    experiment.train_model(
        search_mode=search_mode,
        log_folder=args.logdir,
        search_keys=search_keys,
        config_overrides=config_overrides,
        search_overrides=search_overrides,
        selected_datasets=resolve_dataset_names(experiment.dataset_options, args.datasets),
        selected_options=selected_options,
    )

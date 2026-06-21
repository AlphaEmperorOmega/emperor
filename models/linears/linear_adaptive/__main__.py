from models.parser import get_experiment_parser, resolve_dataset_names, resolve_experiment_mode
from models.linears.linear_adaptive import Experiment, ExperimentPreset

if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentPreset.names(), "models.linears.linear_adaptive")
    args = parser.parse_args()
    mode = resolve_experiment_mode(
        args, ExperimentPreset
    )
    experiment = Experiment(mode.preset)
    experiment.train_model(
        search_mode=mode.search_mode,
        log_folder=args.logdir,
        search_keys=mode.search_keys,
        config_overrides=mode.config_overrides,
        search_overrides=mode.search_overrides,
        selected_datasets=resolve_dataset_names(experiment.dataset_options, args.datasets),
        selected_presets=mode.selected_presets,
        callbacks=mode.monitor_callbacks,
    )

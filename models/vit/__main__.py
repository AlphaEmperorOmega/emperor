from models.parser import get_experiment_parser
from models.vit import Experiment, ExperimentOptions

if __name__ == "__main__":
    parser = get_experiment_parser(ExperimentOptions.names())
    args = parser.parse_args()
    config_option = ExperimentOptions.get_option(args.name)
    experiment = Experiment(config_option)
    experiment.train_model(num_samples=args.num_samples, log_folder=args.log_folder)

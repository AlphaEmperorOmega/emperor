from models.package_cli import run_model_package_cli
from models.vit.expert_linear_adaptive import Experiment, ExperimentPreset

EXPERIMENT_MODULE_PATH = "models.vit.expert_linear_adaptive"

if __name__ == "__main__":
    run_model_package_cli(
        experiment_type=Experiment,
        preset_type=ExperimentPreset,
        module_path=EXPERIMENT_MODULE_PATH,
    )

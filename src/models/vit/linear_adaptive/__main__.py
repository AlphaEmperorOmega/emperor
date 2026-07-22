from models.package_cli import run_model_package_cli

MODEL_PACKAGE_KEY = "vit/linear_adaptive"


if __name__ == "__main__":
    raise SystemExit(run_model_package_cli(MODEL_PACKAGE_KEY))

# Compare Training Runs by Default

The Viewer Compare workspace will become a **Training Run Comparison** surface by default: users compare selected historical **Training Runs** through compact scalar graphs and metric summaries, optionally filtered by **Log Experiment**. The existing model and preset configuration comparison will remain as a secondary Configs mode during the transition, because replacing it outright would remove a useful static diff workflow while leaving Compare as a model-configuration tool would conflict with the user's need to inspect historical training results quickly.

The comparison unit is a historical **Training Run**, not a **Log Experiment**; Log Experiments are used for filtering/grouping and quick-add workflows. Compare remains read-only, historical-log based, independent from Logs workspace selection state, and capped at eight selected Training Runs so overlaid charts and summary tables stay readable.

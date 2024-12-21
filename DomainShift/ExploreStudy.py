import optuna
from optuna.visualization import plot_optimization_history

# Study organization
storage_url = "sqlite:///optuna_study.db"
study_name = 'cartpole_study_DSP_Random'

# Create a new study or load an existing study with a pruner
pruner = optuna.pruners.PercentilePruner(99)
study = optuna.create_study(study_name=study_name, storage=storage_url, direction='maximize', load_if_exists=True, pruner=pruner)

# After optimization, access the best trial
best_trial = study.best_trial

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
print(" Value: ", best_trial.value)
print(" Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Plot the optimization history of the study
plot = plot_optimization_history(study)

# Save the plotly figure to a file
plot.write_image('optimization_history.png')

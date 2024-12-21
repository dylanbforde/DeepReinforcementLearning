import optuna

# study organisation
storage_url = "sqlite:///optuna_study.db"
study_name = 'cartpole_study_DSP'

# Load the study
study = optuna.load_study(study_name=study_name, storage=storage_url)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
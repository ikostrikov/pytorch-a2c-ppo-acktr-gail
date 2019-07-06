from comet_ml import API
import numpy as np
api = API()
exp_ids = {"Hyrule-Mini-All-Shaped-v1": "mweiss17/navi-corl-2019/164622b85b0645de8bfbecea9ae249c6"}

for name, exp_id in exp_ids.items():
    experiment = api.get()
    parameters = api.get_experiment_parameters(exp_id)
    print(parameters)
    reward_mean = np.array(experiment.metrics_raw["Reward Mean"])
    reward_max = np.array( experiment.metrics_raw["Reward Max"])
    reward_min = np.array(experiment.metrics_raw["Reward Min"])
    episode_length_max = np.array(experiment.metrics_raw["Episode Length Max"])
    episode_length_min = np.array(experiment.metrics_raw["Episode Length Min"])
    episode_length_mean = experiment.metrics_raw["Episode Length Mean"]
    episodic_success_rate = experiment.metrics_raw["Episodic Success Rate"]
    loss = experiment.metrics_raw["loss"]

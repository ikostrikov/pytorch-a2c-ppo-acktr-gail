from _warnings import warn

from comet_ml import API
import numpy as np
import matplotlib.pyplot as plt


api = API()
# exp_ids = {"Hyrule-Mini-All-Shaped-v1": "mweiss17/navi-corl-2019/164622b85b0645de8bfbecea9ae249c6", "pacman": "mweiss17/navi-corl-2019/40be0457f4174377963abe5da282a944"}

# 10 procs
exp_ids = {
    "Hyrule-Mini-All-Shaped-v1-s0-10p": "mweiss17/navi-corl-2019/164622b85b0645de8bfbecea9ae249c6",
    "Hyrule-Mini-All-Shaped-v1-s1-10p": "mweiss17/navi-corl-2019/5b7fe5ca238142f6a57d00319fafeec5",
    "Hyrule-Mini-NoImg-Shaped-v1-s0-10p": "mweiss17/navi-corl-2019/a2ef36e62bcd41098d5b0a1994913268",
    "Hyrule-Mini-NoImg-Shaped-v1-s1-10p": "mweiss17/navi-corl-2019/2da12b34c1eb4f4c9f2cdd6f0ec867af",
    "Hyrule-Mini-NoGPS-Shaped-v1-s0-10p": "mweiss17/navi-corl-2019/44daa65b9fde4336a0049a9efce7610c",
    "Hyrule-Mini-NoGPS-Shaped-v1-s1-10p": "mweiss17/navi-corl-2019/fcaf64ee63ab4fa19c58e0578e70f1b4",
    "Hyrule-Mini-ImgOnly-Shaped-v1-s6-10p": "mweiss17/navi-corl-2019/2b960c6b3db24e6fb859c71f71a5a58b",
    "Hyrule-Mini-ImgOnly-Shaped-v1-s7-10p": "mweiss17/navi-corl-2019/5dfab8d58a524ab3a443d1a8d46e9d99",
}

plot_info = {
    "Hyrule-Mini-All-Shaped-v1": {'color': '#22a784', 'plot_name': 'AllObs'},
    "Hyrule-Mini-NoImg-Shaped-v1": {'color': '#fde724', 'plot_name': 'NoImg'},
    "Hyrule-Mini-NoGPS-Shaped-v1": {'color': '#440154', 'plot_name': 'NoGPS'},
    "Hyrule-Mini-ImgOnly-Shaped-v1": {'color': '#29788e', 'plot_name': 'ImgOnly'},
}

reported_metrics = ["Reward Mean", "Episodic Success Rate" , "Episode Length Mean ",]


def build_plot_dict(orig_env_name, raw_tuples, final_data, log_ts):
    for i in range(len(raw_tuples)):

        env_name = orig_env_name + ' - ' + reported_metrics[i]
        temp_data = np.array([list(x) for x in raw_tuples[i]]).transpose()

        # Preprocess for equal log intervals
        data = np.zeros((temp_data.shape[0], len(log_ts)))
        data[0, :] = log_ts
        for j in range(len(log_ts)):
            index = np.where(temp_data[0] == log_ts[j])[0]
            if index.size == 0:
                continue
            data[:, j] = temp_data[:, index[0]]

        # If same experiment with different seed
        if env_name in final_data:
            final_data[env_name]['n'] += 1
            data_concat = np.vstack((final_data[env_name]['data'], data[1, :]))
            final_data[env_name]['data'] = data_concat

        # If first experiment with these hyperparams.
        else:
            final_data[env_name] = {'metric': reported_metrics[i], 'data': data, 'n': 1}

    return final_data


final_data = {}
for name, exp_id in exp_ids.items():
    experiment = api.get(exp_id)
    for d in experiment.parameters:
        if d["name"] == 'env-name':
            assert d['valueMax'] in name
            env_name = d['valueMax']
        if d["name"] == 'log_interval':
            log_int = d['valueMax']

    reward_mean = experiment.metrics_raw[reported_metrics[0]]
    episodic_success_rate = experiment.metrics_raw[reported_metrics[1]]
    episode_length_mean = experiment.metrics_raw[reported_metrics[2]]
    if name == 'Hyrule-Mini-All-Shaped-v1-s0-10p':
        logged_timesteps = np.array([list(x) for x in reward_mean]).transpose()[0]


    final_data = build_plot_dict(orig_env_name=env_name,
                                 raw_tuples=[reward_mean, episodic_success_rate, episode_length_mean],
                                 final_data=final_data,
                                 log_ts=logged_timesteps)


for metric in reported_metrics:
    fig = plt.figure()

    plt.title(metric, fontsize=14)
    plt.xlabel('Timesteps', fontsize=10)
    plt.ylabel(metric, fontsize=10)

    for key, val in final_data.items():
        if metric in key:
            color = plot_info[key.replace(' - ' + metric, "")]['color']
            label = plot_info[key.replace(' - ' + metric, "")]['plot_name']

            met_mean = np.mean(val['data'][1:], axis=0)
            met_std = np.std(val['data'][1:], axis=0)

            plt.fill_between(val['data'][0], met_mean - met_std, met_mean + met_std, alpha=0.1,
                             color=color)
            plt.plot(val['data'][0], met_mean, color, label=label)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend()
    plt.savefig(metric + ".png")


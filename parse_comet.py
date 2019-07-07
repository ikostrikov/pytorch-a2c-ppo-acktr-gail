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
}

colors = {
    "Hyrule-Mini-All-Shaped-v1": '#000c1a',
    "Hyrule-Mini-NoImg-Shaped-v1": '#6b0000',
    "Hyrule-Mini-NoGPS-Shaped-v1": '#0d8c00',
    "Hyrule-Mini-ImgOnly-Shaped-v1": '#000cda',
    "Hyrule-Mini-Pacman-v1": '#e23f3a'
}

reported_metrics = ["Reward Mean", "Episodic Success Rate"]  # , "Episode Length Mean",]


def build_plot_dict(orig_env_name, raw_tuples, final_data, log_ts):
    for i in range(len(raw_tuples)):

        env_name = orig_env_name + ' - ' + reported_metrics[i]
        temp_data = np.array([list(x) for x in raw_tuples[i]]).transpose()

        # Preprocess for equal log intervals
        data = np.zeros((temp_data.shape[0], len(log_ts)))
        for j in range(len(log_ts)):
            index = np.where(temp_data[0] == log_ts[j])[0]
            if index.size == 0:
                continue
            data[:, j] = temp_data[:, index[0]]

        # If same experiment with different seed
        if env_name in final_data:
            final_data[env_name]['n'] += 1
            new_data = (final_data[env_name]['data'] + data) / final_data[env_name]['n']
            final_data[env_name]['data'] = new_data

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
    # episode_length_mean = np.array(experiment.metrics_raw[reported_metrics[1]])
    # print(episode_length_mean)
    if name == 'Hyrule-Mini-All-Shaped-v1-s0-10p':
        logged_timesteps = np.array([list(x) for x in reward_mean]).transpose()[0]

    # else:
    final_data = build_plot_dict(orig_env_name=env_name,
                                 raw_tuples=[reward_mean, episodic_success_rate],
                                 final_data=final_data,
                                 log_ts=logged_timesteps)


for metric in reported_metrics:
    fig = plt.figure()

    plt.title(metric, fontsize=14)
    plt.xlabel('Timesteps', fontsize=10)
    plt.ylabel(metric, fontsize=10)

    for key, val in final_data.items():
        if metric in key:
            plt.plot(val['data'][0], val['data'][1], colors[key.replace(' - '+metric, "")], label=key)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend()
    plt.savefig(metric + ".png")


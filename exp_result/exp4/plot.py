import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

metrics = ['imp', 'mean']
params = ['Rx', 'Ry', 'T', 'D']
for metric in metrics:
    all_data = {}
    all_metric = {}
    for param in params:
        all_data[param] = pd.read_csv(f'{param}.csv')
        all_metric[param] = {}
        if param == 'Rx':
            all_metric[param]['names'] = ['0.25','0.5','0.75','1.0','1.25', '1.5']
        elif param == 'Ry':
            all_metric[param]['names'] = ['0.25','0.5','0.75','1.0','1.25', '1.5']
        elif param == 'D':
            all_metric[param]['names'] = ['0.05','0.1','0.15','0.2','0.25']
        elif param == 'T':
            all_metric[param]['names'] = ['0.05','0.1','0.15','0.2','0.25']
        all_metric[param]['no_anchor_single'] = []
        all_metric[param]['gt_single'] = []
        all_metric[param]['four_anchor_single'] = []
        all_metric[param]['ten_anchor_single'] = []
        all_metric[param]['no_anchor_multi'] = []
        all_metric[param]['gt_multi'] = []
        all_metric[param]['four_anchor_multi'] = []
        all_metric[param]['ten_anchor_multi'] = []

        no_anchor_single = []
        gt_single = []
        four_anchor_single = []
        ten_anchor_single = []
        no_anchor_multi = []
        gt_multi = []
        four_anchor_multi = []
        ten_anchor_multi = []

        all_metric[param]['no_anchor_single'].append(all_data[param][all_data[param]['name']=="0.0_0.0_0.0_0.0_0.0_0.0"][f'single_no_anchor_dist_{metric}'].values[0])
        all_metric[param]['gt_single'].append(all_data[param][all_data[param]['name']=="0.0_0.0_0.0_0.0_0.0_0.0"][f'single_no_anchor_gt_dist_{metric}'].values[0])
        all_metric[param]['four_anchor_single'].append(eval(all_data[param][all_data[param]['name']=="0.0_0.0_0.0_0.0_0.0_0.0"][f'single_anchor_dist_{metric}'].values[0])[0])
        all_metric[param]['ten_anchor_single'].append(eval(all_data[param][all_data[param]['name']=="0.0_0.0_0.0_0.0_0.0_0.0"][f'single_anchor_dist_{metric}'].values[0])[1])

        all_metric[param]['no_anchor_multi'].append(all_data[param][all_data[param]['name']=="0.0_0.0_0.0_0.0_0.0_0.0"][f'multi_no_anchor_dist_{metric}'].values[0])
        all_metric[param]['gt_multi'].append(all_data[param][all_data[param]['name']=="0.0_0.0_0.0_0.0_0.0_0.0"][f'multi_no_anchor_gt_dist_{metric}'].values[0])
        all_metric[param]['four_anchor_multi'].append(eval(all_data[param][all_data[param]['name']=="0.0_0.0_0.0_0.0_0.0_0.0"][f'multi_anchor_dist_{metric}'].values[0])[0])
        all_metric[param]['ten_anchor_multi'].append(eval(all_data[param][all_data[param]['name']=="0.0_0.0_0.0_0.0_0.0_0.0"][f'multi_anchor_dist_{metric}'].values[0])[1])

        for name in all_metric[param]['names']:
            if param == 'Rx':
                combine1 = f'{name}_0.0_0.0_0.0_0.0_0.0'
                combine2 = f'-{name}_0.0_0.0_0.0_0.0_0.0'
            elif param == 'Ry':
                combine1 = f'0.0_{name}_0.0_0.0_0.0_0.0'
                combine2 = f'0.0_-{name}_0.0_0.0_0.0_0.0'
            elif param == 'T':
                combine1 = f'0.0_0.0_0.0_{name}_0.0_0.0'
                combine2 = f'0.0_0.0_0.0_-{name}_0.0_0.0'
            elif param == 'D':
                combine1 = f'0.0_0.0_0.0_0.0_0.0_{name}'
                combine2 = f'0.0_0.0_0.0_0.0_0.0_-{name}'
            all_metric[param]['no_anchor_single'].append((all_data[param][all_data[param]['name']==combine1][f'single_no_anchor_dist_{metric}'].values[0] + all_data[param][all_data[param]['name']==combine2][f'single_no_anchor_dist_{metric}'].values[0]) / 2)

            all_metric[param]['gt_single'].append((all_data[param][all_data[param]['name']==combine1][f'single_no_anchor_gt_dist_{metric}'].values[0] + all_data[param][all_data[param]['name']==combine2][f'single_no_anchor_gt_dist_{metric}'].values[0]) / 2)
            all_metric[param]['four_anchor_single'].append((eval(all_data[param][all_data[param]['name']==combine1][f'single_anchor_dist_{metric}'].values[0])[0] + eval(all_data[param][all_data[param]['name']==combine2][f'single_anchor_dist_{metric}'].values[0])[0]) / 2)
            all_metric[param]['ten_anchor_single'].append((eval(all_data[param][all_data[param]['name']==combine1][f'single_anchor_dist_{metric}'].values[0])[1] + eval(all_data[param][all_data[param]['name']==combine2][f'single_anchor_dist_{metric}'].values[0])[1]) / 2)

            all_metric[param]['no_anchor_multi'].append((all_data[param][all_data[param]['name']==combine1][f'multi_no_anchor_dist_{metric}'].values[0] + all_data[param][all_data[param]['name']==combine2][f'multi_no_anchor_dist_{metric}'].values[0]) / 2)
            all_metric[param]['gt_multi'].append((all_data[param][all_data[param]['name']==combine1][f'multi_no_anchor_gt_dist_{metric}'].values[0] + all_data[param][all_data[param]['name']==combine2][f'multi_no_anchor_gt_dist_{metric}'].values[0]) / 2)
            all_metric[param]['four_anchor_multi'].append((eval(all_data[param][all_data[param]['name']==combine1][f'multi_anchor_dist_{metric}'].values[0])[0] + eval(all_data[param][all_data[param]['name']==combine2][f'multi_anchor_dist_{metric}'].values[0])[0]) / 2)
            all_metric[param]['ten_anchor_multi'].append((eval(all_data[param][all_data[param]['name']==combine1][f'multi_anchor_dist_{metric}'].values[0])[1] + eval(all_data[param][all_data[param]['name']==combine2][f'multi_anchor_dist_{metric}'].values[0])[1]) / 2)

    fig, ax_arr = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(12,6))
    # matplotlib.rcParams['font.family'] = 'Times New Roman'

    for i in range(4):
        if params[i] == 'Rx' or params[i] == 'Ry':
            errors = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
        elif params[i] == 'T' or params[i] == 'D':
            errors = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        ax_arr[0, i].plot(errors, all_metric[params[i]]['no_anchor_single'], 's--', label='No Anchor', color= '#1f77b4', markersize=5, linewidth=1)
        ax_arr[0, i].plot(errors, all_metric[params[i]]['four_anchor_single'], '^--', label='4 Anchors', color= '#ff7f0e', markersize=5, linewidth=1)
        ax_arr[0, i].plot(errors, all_metric[params[i]]['ten_anchor_single'], '>--', label='10 Anchors', color= '#2ca02c', markersize=5, linewidth=1)
        ax_arr[0, i].plot(errors, all_metric[params[i]]['gt_single'], 'x--', label='Ground Truth Params', color= '#d62728', markersize=5, linewidth=1)
        if i == 0:
            if metric == 'imp':
                ax_arr[0, i].set_ylabel('Improvement Ratio (%)\n for Single-Camera Localization')
            else:
                ax_arr[0, i].set_ylabel('Average Distance (m)\n for Single-Camera Localization')
        if i == 3 and metric == 'mean':
            ax_inset = inset_axes(ax_arr[0,i], width="70%", height="30%", loc=9, bbox_to_anchor=[0,0,1,0.5], bbox_transform=ax_arr[0,i].transAxes)
            ax_inset.plot(errors, all_metric[params[i]]['no_anchor_single'], 's--', label='No Anchor', color= '#1f77b4', markersize=5, linewidth=1)
            ax_inset.plot(errors, all_metric[params[i]]['four_anchor_single'], '^--', label='4 Anchors', color= '#ff7f0e', markersize=5, linewidth=1)
            ax_inset.plot(errors, all_metric[params[i]]['ten_anchor_single'], '>--', label='10 Anchors', color= '#2ca02c', markersize=5, linewidth=1)
            ax_inset.plot(errors, all_metric[params[i]]['gt_single'], 'x--', label='Ground Truth Params', color= '#d62728', markersize=5, linewidth=1)
            ax_inset.set_xlim(0, 0.25)
            ax_inset.set_ylim(0.408, 0.412)
            ax_inset.set_xticklabels('')
            ax_inset.set_yticklabels('')
            mark_inset(ax_arr[0,i], ax_inset, loc1=1, loc2=3, fc="none", ec='0.5')

    for i in range(4):
        if params[i] == 'Rx' or params[i] == 'Ry':
            errors = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
        elif params[i] == 'T' or params[i] == 'D':
            errors = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
        ax_arr[1, i].plot(errors, all_metric[params[i]]['no_anchor_multi'], 's--', label='No Anchor', color= '#1f77b4', markersize=5, linewidth=1)
        ax_arr[1, i].plot(errors, all_metric[params[i]]['four_anchor_multi'], '^--', label='4 Anchors', color= '#ff7f0e', markersize=5, linewidth=1)
        ax_arr[1, i].plot(errors, all_metric[params[i]]['ten_anchor_multi'], '>--', label='10 Anchors', color= '#2ca02c', markersize=5, linewidth=1)
        ax_arr[1, i].plot(errors, all_metric[params[i]]['gt_multi'], 'x--', label='Ground Truth Params', color= '#d62728', markersize=5, linewidth=1)
        if i == 0:
            if metric == 'imp':
                ax_arr[1, i].set_ylabel('Improvement Ratio (%)\n for Multi-Camera Localization')
            else:
                ax_arr[1, i].set_ylabel('Average Distance (m)\n for Multi-Camera Localization')

    for i in range(4):
        ax_arr[1, i].set_xlabel(f'{params[i]} error')

    for idx, ax in enumerate(ax_arr.flatten()):
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        

    labels = ['No Anchor', '4 Anchors', '10 Anchors', 'Ground Truth Params']
    markers = ['s', '^', '>', '+']
    linestyle = '--'
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    lines = [plt.Line2D([0], [0], color=color, linestyle=linestyle, marker=marker, linewidth=1, markersize=5) for marker, color in zip(markers, colors)]
    if metric == 'imp':
        fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 0.75), ncol=1)
    else:
        fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1), ncol=1)


    plt.tight_layout()

    plt.savefig(f'Fig4-{metric}.pdf')

    plt.show()
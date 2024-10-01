import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

other_ = ["0.1_0.1_0.0_0.1_0.0_0.1","0.2_0.2_0.0_0.1_0.0_0.1","0.3_0.3_0.0_0.15_0.0_0.15"]
params = ['imp', 'mean']
for param in params:
    fig, ax_arr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12,6))
    for i in range(3):
        df = pd.read_csv(f'set{i}.csv')
        
        purturb = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        purturb = [str(p) for p in purturb]
        no_anchor = []
        four_anchor = []
        ten_anchor = []
        gt_ = []

        multi_no_anchor = []
        multi_four_anchor = []
        multi_ten_anchor = []
        multi_gt_ = []
        for p in purturb:
            no_anchor.append(df[df['name']==f'{other_[i]}_{p}'][f'single_no_anchor_dist_{param}'].values[0])
            four_anchor.append(eval(df[df['name']==f'{other_[i]}_{p}'][f'single_anchor_dist_{param}'].values[0])[0])
            ten_anchor.append(eval(df[df['name']==f'{other_[i]}_{p}'][f'single_anchor_dist_{param}'].values[0])[-1])
            gt_.append(df[df['name']==f'{other_[i]}_{p}'][f'single_no_anchor_gt_dist_{param}'].values[0])

            multi_no_anchor.append(df[df['name']==f'{other_[i]}_{p}'][f'multi_no_anchor_dist_{param}'].values[0])
            multi_four_anchor.append(eval(df[df['name']==f'{other_[i]}_{p}'][f'multi_anchor_dist_{param}'].values[0])[0])
            multi_ten_anchor.append(eval(df[df['name']==f'{other_[i]}_{p}'][f'multi_anchor_dist_{param}'].values[0])[-1])
            multi_gt_.append(df[df['name']==f'{other_[i]}_{p}'][f'multi_no_anchor_gt_dist_{param}'].values[0])

        ax_arr[0, i].plot(purturb, no_anchor, 's--', label='No Anchor', color= '#1f77b4', markersize=5, linewidth=1)
        ax_arr[0, i].plot(purturb, four_anchor, '^--', label='4 Anchors', color= '#ff7f0e', markersize=5, linewidth=1)
        ax_arr[0, i].plot(purturb, ten_anchor, '>--', label='10 Anchors', color= '#2ca02c', markersize=5, linewidth=1)
        ax_arr[0, i].plot(purturb, gt_, 'x--', label='Ground Truth Params', color= '#d62728', markersize=5, linewidth=1)

        ax_arr[1, i].plot(purturb, multi_no_anchor, 's--', label='No Anchor', color= '#1f77b4', markersize=5, linewidth=1)
        ax_arr[1, i].plot(purturb, multi_four_anchor, '^--', label='4 Anchors', color= '#ff7f0e', markersize=5, linewidth=1)
        ax_arr[1, i].plot(purturb, multi_ten_anchor, '>--', label='10 Anchors', color= '#2ca02c', markersize=5, linewidth=1)
        ax_arr[1, i].plot(purturb, multi_gt_, 'x--', label='Ground Truth Params', color= '#d62728', markersize=5, linewidth=1)

        ax_arr[1, i].set_xlabel(f'Pix perturb (Parameter Set {i+1})')

    for idx, ax in enumerate(ax_arr.flatten()):
        ax.grid(color='gray', linestyle='--', linewidth=0.5)

    if param == 'mean':
        ax_arr[0, 0].set_ylabel('Average Distance (m)\n for Single-Camera Localization')
        ax_arr[1, 0].set_ylabel('Average Distance (m)\n for Multi-Camera Localization')
    else:
        ax_arr[0, 0].set_ylabel('Improvement Ratio (%)\n for Single-Camera Localization')
        ax_arr[1, 0].set_ylabel('Improvement Ratio (%)\n for Multi-Camera Localization')

    labels = ['No Anchor', '4 Anchors', '10 Anchors', 'Ground Truth Params']
    markers = ['s', '^', '>', '+']
    linestyle = '--'
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    lines = [plt.Line2D([0], [0], color=color, linestyle=linestyle, marker=marker, linewidth=1, markersize=5) for marker, color in zip(markers, colors)]
    if param == 'imp':
        fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
    else:
        fig.legend(lines, labels, loc='upper left',  bbox_to_anchor=(0.075, 0.97),ncol=1)
    plt.tight_layout()

    plt.savefig(f'Fig5_{param}.pdf')

    plt.show()


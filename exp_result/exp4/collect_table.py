import pandas as pd

type_ = ['Rx', 'Ry','T','D']
pos = [0,1,3,-1]
camera_type = ['multi', 'single']
param = ['mean', 'imp']

for p in param:
    big_dic = {'multi':{'error_type':[], 'error':[], 'zero':[], 'four':[],'six':[],'eight':[],'ten':[], 'gt':[]},
            'single':{'error_type':[], 'error':[], 'zero':[], 'four':[],'six':[],'eight':[],'ten':[], 'gt':[]}}
    for idx,error in enumerate(type_):
        df = pd.read_csv(f"{error}.csv")
        df['error'] = df['name'].apply(lambda x: x.split('_')[pos[idx]])
        df.sort_values(by='error',inplace=True)

        df['multi_four_anchor'] = df[f'multi_anchor_dist_{p}'].apply(lambda x: eval(x)[0])
        df['multi_six_anchor'] = df[f'multi_anchor_dist_{p}'].apply(lambda x: eval(x)[1])
        df['multi_eight_anchor'] = df[f'multi_anchor_dist_{p}'].apply(lambda x: eval(x)[2])
        df['multi_ten_anchor'] = df[f'multi_anchor_dist_{p}'].apply(lambda x: eval(x)[3])

        df['single_four_anchor'] = df[f'single_anchor_dist_{p}'].apply(lambda x: eval(x)[0])
        df['single_six_anchor'] = df[f'single_anchor_dist_{p}'].apply(lambda x: eval(x)[1])
        df['single_eight_anchor'] = df[f'single_anchor_dist_{p}'].apply(lambda x: eval(x)[2])
        df['single_ten_anchor'] = df[f'single_anchor_dist_{p}'].apply(lambda x: eval(x)[3])

        for camera in camera_type:
            big_dic[camera]['error_type'].extend([error]*len(df))
            big_dic[camera]['error'].extend(df['error'].values)
            big_dic[camera]['zero'].extend(df[f'{camera}_no_anchor_dist_{p}'].values)
            big_dic[camera]['four'].extend(df[f'{camera}_four_anchor'].values)
            big_dic[camera]['six'].extend(df[f'{camera}_six_anchor'].values)
            big_dic[camera]['eight'].extend(df[f'{camera}_eight_anchor'].values)
            big_dic[camera]['ten'].extend(df[f'{camera}_ten_anchor'].values)
            big_dic[camera]['gt'].extend(df[f'{camera}_no_anchor_gt_dist_{p}'].values)

    multi = pd.DataFrame(big_dic['multi'])
    single = pd.DataFrame(big_dic['single'])

    multi.to_csv(f'multi_{p}.csv',index=False)
    single.to_csv(f'single_{p}.csv',index=False)

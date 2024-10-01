cd ..
rx_errors=(0.1 0.1 0.1 0.1 0.1 0.1 0.2 0.2 0.2 0.2 0.2 0.2 0.3 0.3 0.3 0.3 0.3 0.3)
ry_errors=(0.1 0.1 0.1 0.1 0.1 0.1 0.2 0.2 0.2 0.2 0.2 0.2 0.3 0.3 0.3 0.3 0.3 0.3)
T_errors=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.15 0.15 0.15 0.15 0.15 0.15)
D_errors=(0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.15 0.15 0.15 0.15 0.15 0.15)
pix_perturb_levels=(5 10 15 20 25 30 5 10 15 20 25 30 5 10 15 20 25 30)
for i in {0..17}
do
    python main.py --rx_error ${rx_errors[$i]} --ry_error ${ry_errors[$i]} --rz_error 0.0 --T_error ${T_errors[$i]} --D_error ${D_errors[$i]} --K_error 0.0 --pix_perturb_level ${pix_perturb_levels[$i]} --task perturb --test_data large_virtual_sample
done

cd ./exp_result/exp5
python exp5_all_process.py
python plot.py  # Get figure 5 and appendix figure
python collect_table.py # Get appendix table
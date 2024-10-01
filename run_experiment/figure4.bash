cd ..
rxy_error_values="0.0 0.25 0.5 0.75 1.0 1.25 1.5 -0.25 -0.5 -0.75 -1.0 -1.25 -1.5"
for rxy_error in $rxy_error_values
do
    python main.py --rx_error $rxy_error --ry_error 0.0 --rz_error 0.0 --T_error 0.0 --D_error 0.0 --K_error 0.0 --task error_distance --test_data large_virtual_sample
    python main.py --rx_error 0.0 --ry_error $rxy_error --rz_error 0.0 --T_error 0.0 --D_error 0.0 --K_error 0.0 --task error_distance --test_data large_virtual_sample
done

td_error_values="0.0 0.05 0.1 0.15 0.2 0.25 -0.05 -0.1 -0.15 -0.2 -0.25"
for td_error in $td_error_values
do
    python main.py --rx_error 0.0 --ry_error 0.0 --rz_error 0.0 --T_error $td_error --K_error 0.0 --D_error 0.0 --task error_distance --test_data large_virtual_sample
    python main.py --rx_error 0.0 --ry_error 0.0 --rz_error 0.0 --T_error 0.0 --K_error 0.0 --D_error $td_error --task error_distance --test_data large_virtual_sample
done

cd ./exp_result/exp4
python exp4_all_process.py
python plot.py  # Get figure 4 and appendix figure
python collect_table.py # Get appendix table
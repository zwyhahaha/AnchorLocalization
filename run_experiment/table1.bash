cd ..
rxy_errors=(0.25 0.5 0.75 1.0 1.25 1.5)
td_errors=(0.05 0.1 0.15 0.2 0.25)

for i in {0..5}
do
    python main.py --rx_error ${rxy_errors[$i]} --ry_error 0.0 --rz_error 0.0 --T_error 0.0 --D_error 0.0 --K_error 0.0 --task check_angular_error --test_data large_virtual_sample
done

for i in {0..5}
do
    python main.py --rx_error 0.0 --ry_error ${rxy_errors[$i]} --rz_error 0.0 --T_error 0.0 --D_error 0.0 --K_error 0.0 --task check_angular_error --test_data large_virtual_sample
done

for i in {0..4}
do
    python main.py --rx_error 0.0 --ry_error 0.0 --rz_error 0.0 --T_error ${td_errors[$i]} --D_error 0.0 --K_error 0.0 --task check_angular_error --test_data large_virtual_sample   
done

for i in {0..4}
do
    python main.py --rx_error 0.0 --ry_error 0.0 --rz_error 0.0 --T_error 0.0 --D_error 0.0 --K_error ${td_errors[$i]} --task check_angular_error --test_data large_virtual_sample
done
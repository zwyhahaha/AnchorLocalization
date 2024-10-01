cd ..
python main.py --batch_size 1 --test_data real --task batch_experiment
python main.py --batch_size 3 --test_data real --task batch_experiment
python main.py --batch_size 5 --test_data real --task batch_experiment
python main.py --batch_size 7 --test_data real --task batch_experiment

cd exp_result/exp6
python collect.py
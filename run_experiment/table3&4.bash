cd ..
python main.py --use_anchor True --det_type head --test_data real --task representive_point
python main.py --use_anchor True --det_type foot --test_data real --task representive_point
python main.py --use_anchor False --det_type head --test_data real --task representive_point
python main.py --use_anchor False --det_type foot --test_data real --task representive_point

cd ./exp_result/exp4
python collect.py
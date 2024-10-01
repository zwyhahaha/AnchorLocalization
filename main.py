from simu_exp import perturb_test, real_exp1, real_exp2, simu_exp1, simu_exp2
from utils.config import get_args
import os
import pickle
import utils.camera_data_serielize_fixparams as camera_data_serielize_fixparams

args = get_args()

error_dict = {
    'rx_error': args.rx_error,
    'ry_error': args.ry_error,
    'rz_error': args.rz_error,
    'T_error': args.T_error,
    'K_error': args.K_error,
    'D_error': args.D_error
}
error_comb = "_".join([f"{param}{value}" for param, value in zip(error_dict.keys(), error_dict.values())])

if 'real' in args.test_data: # Experiment under real case
    # Load data
    test_path = "data/real"
    args.camera_dir = os.path.join(test_path, "monitors_json")
    args.nc = 6
    args.cams = ['C142c79d90039ce7a', 'C78fcb895887a3935', 'Ccf4692cb9838d53c', 'C58ea83521a4169a3', 'C9f859b0853a4bfe0', 'Cf9ed46a5770558d3']
    args.use_distort = True

    with open("data/real/features/all-df1700.pkl", 'rb') as file:
        trace_data = pickle.load(file)

    args.t1 = 0
    args.t2 = 1000
    trace_data = trace_data[(trace_data['findex'] >= args.t1) & (trace_data['findex'] <= args.t2)]

    # Run experiment
    if args.task == "representive_point" or args.task == "representive_point_multi":
        real_exp1.main(args, trace_data) # Table 3 and Table 4
    elif args.task == "batch_experiment":
        real_exp2.main(args, trace_data) # Table 5

elif 'virtual' in args.test_data: # Experiment under virtual case
    # Load data
    test_path = "data/large"
    args.nc = 47
    args.camera_dir = os.path.join("monitors_json_large", error_comb)
    args.cams = ["C513d1638af592626", "Cbb26f32b89238d17", "Cac5d12eb25394fe3", "C50f7409efbb0e13f", "C9c64e171827659d7", "C236f956c4eeddc52", "C9fcaf220eeec07bf", "Ce642d79a519aead2", "Ce794ec090de01c1c", "Cfce7ac3bae745f04", "Cf549692b2082d16b", "C9a3e5ffa338531bf", "C55df5f7a1c8a4c17", "Ca8a8817c8097bc03", "Ce16104a95c9ac266", "C469f539fa15bcbfe", "C89ae456a1aedfea8", "C7b0a2e82f162cb6a", "C8314228e0241282f", "C699c5c55502856cb", "C98e405a83663ab4a", "C003a3f8246d78c8a", "Cef51ac47b946e535", "Cd49f9de790ba8f14", "C3151ce23b866fb8a", "C255a40d57d3e132d", "Cab842a9dc6c6918c", "C6b957d8d29bfd9d4", "C2ea85e7b02daa189", "C2af59912e43b0cc2", "C825bb203cf76766e", "Ce61a8ac41383f178", "Cbed16c2738d6f6c4", "C9f03f31b2a7afd7e", "C2b318a6642bf6276", "Ca8e0c404035c3548", "C39925050c3b34a6f", "C2deb4973b432cea7", "C731db75b066ad72a", "Cde1d44f5db8ad3ec", "Ca988ffd85e1340b6", "Cfb3d8da71ac01e82", "Cfba6a0886b38f6a7", "C0e3b602043ced66e", "Ccdb30527ec9684f9", "C28b7933eaa1e99d1", "Ca5bd6d5a53ae782b"]

    with open("data/large-virtual/features/sample-df.pkl", 'rb') as file:
        trace_data = pickle.load(file)

    args.camera_dir = os.path.join("monitors_json_large", error_comb)
    args.t1 = 3000
    args.t2 = 4000
    trace_data = trace_data[(trace_data['findex'] >= args.t1) & (trace_data['findex'] <= args.t2)]

    if not os.path.exists(args.camera_dir):
        os.mkdir(args.camera_dir)
        camera_data_serielize_fixparams.do(args, "./data/large-virtual", args.camera_dir)
    
    # Run experiment
    if args.task == "check_angular_error":
        perturb_test.main(args) # Table 1 experiment
    elif args.task == "error_distance":
        simu_exp1.main(args, trace_data) # Figure 4: under different camera parameter errors
    elif args.task == "perturb":
        simu_exp2.main(args, trace_data) # Figure 5: under different pixel perturbations
else:
    raise NotImplementedError
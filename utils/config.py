import argparse

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        pass 

def get_args():
    parser = argparse.ArgumentParser(description="Configuration settings")

    valid_test_data = ["large_virtual_sample", "real"]
    parser.add_argument("--test_data", default="real", type=str, choices=valid_test_data, help="choose test data")
    parser.add_argument("--camera_dir", default="", type=str, help="cam param direction")
    parser.add_argument("--task", default="batch_experiment", type=str, help="cam param direction")

    # Error parameters
    parser.add_argument("--error", default=0, type=float, help="Error value")
    parser.add_argument("--rx_error", default=2.3, type=float, help="R error for rx")
    parser.add_argument("--ry_error", default=0.4, type=float, help="R error for ry")
    parser.add_argument("--rz_error", default=0.6, type=float, help="R error for rz")
    parser.add_argument("--T_error", default=0.03, type=float, help="T error")
    parser.add_argument("--K_error", default=0.04, type=float, help="K error")
    parser.add_argument("--D_error", default=0.3, type=float, help="D error")

    # Simulation parameters
    parser.add_argument("--na", default=5, type=int, help="number of anchors")
    parser.add_argument("--max_na", default=12, type=int, help="max number of anchors")
    parser.add_argument("--nc", default=None, type=int, help="number of cameras")
    parser.add_argument("--h", default=1.65, type=float, help="H value for initial estimation")
    parser.add_argument("--h_std", default=0.055, type=float, help="H std deviation")
    parser.add_argument("--rho", default=1.00, type=float, help="Rho value")
    parser.add_argument("--batch_rho", default=40, type=float, help="penalty rho for batch localization")
    parser.add_argument("--penalty", default=0.0, type=float, help="(deprecated) Penalty for height deviation")
    parser.add_argument("--t1", default=0, type=int)
    parser.add_argument("--t2", default=1000, type=int)

    # Generate simulation data
    parser.add_argument("--cams", nargs='*', default=[
        "C7ac17b9e0501d6c3", "C7f83c07eaabd5071", "C82c45a8dabf695d1", "C474bc6a0b155cda0",
        "Ce39b560ed4b3c31e", "Ce3538357344da607", "Cf1d53424afe32edc"], help="List of cameras")

    # Localization parameters
    parser.add_argument("--start_method", default='multi_camera_mean', choices=['regression', 'multi_camera_mean'], help="Start method")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--lamda", default=15.0, type=float, help="l2-norm penalty for determine anchor weight w")
    parser.add_argument("--pix_perturb_level", default=3, type=float)
    parser.add_argument("--use_distort", default=True, type=t_or_f, help="whether to use distortion in imaging process")
    parser.add_argument("--det_type", default='head', choices=['head', 'foot'], type=str)
    parser.add_argument("--use_anchor", default=True, type=t_or_f, help="whether to use anchor point")

    args = parser.parse_args()
    return args

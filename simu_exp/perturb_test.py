from utils.camera_simu import Camera


def main(config):
    cam = {}
    for id in config.cams:
        cam[id] = Camera(id, config.na, config.camera_dir, config.use_distort, config)
    
    pix_perturbs = [cam[id].pix_err for id in config.cams]
    pix_perturb = sum(pix_perturbs)/len(pix_perturbs)

    vec_perturbs = [cam[id].vec_err for id in config.cams]
    vec_perturb = sum(vec_perturbs)/len(vec_perturbs)
    
    with open('./exp_result/exp1/table1.txt','a') as f:
        f.write(f"\n{config.rx_error},{config.ry_error},{config.rz_error},{config.T_error},{config.K_error},{config.D_error},{pix_perturb},{vec_perturb}")
    # print(pix_perturb, vec_perturb)

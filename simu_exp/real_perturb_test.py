import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.camera_real import realCamera

rng = np.random.default_rng(10)
pd.options.mode.chained_assignment = None

def main(config):
    
    cam = {}
    id = 'C9f859b0853a4bfe0'
    cam[id] = realCamera(id, config)
    print(id, cam[id].pix_err, cam[id].vec_err)
    
    # Plot the result
    plt.scatter(cam[id].gt_anchor_pix[:, 0], cam[id].gt_anchor_pix[:, 1], color='blue', label='GT Anchor Points')
    plt.scatter(cam[id].prox_anchor_pix[:, 0], cam[id].prox_anchor_pix[:, 1], color='red', label='Prox Anchor Points')

    plt.xlim(0, 1280)
    plt.ylim(0, 720)

    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.title('GT vs. Prox Anchor Points')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()

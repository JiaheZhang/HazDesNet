import cv2
from cv2.ximgproc import guidedFilter
import numpy as np
import matplotlib.pyplot as plt

import model

i = 1
image_path = './images/%d.bmp' % (i)
results_path = './results/%d.png' % (i)

if __name__ == "__main__":
    HazDesNet = model.load_HazDesNet()
    HazDesNet.summary()

    img = cv2.imread(image_path)

    guide = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = np.expand_dims(img, axis=0)
    
    # predict one hazy image
    haz_des_map = HazDesNet.predict(img)

    haz_des_map = haz_des_map[0,:,:,0]

    guide = cv2.resize(guide, (haz_des_map.shape[1], haz_des_map.shape[0]))
    
    haz_des_map = guidedFilter(guide=guide, src=haz_des_map, radius=32, eps=500)

    des_score = np.mean(haz_des_map)

    cv2.imwrite(results_path, haz_des_map*255)

    print("the haze density score of " + image_path + " is: %.2f" % (des_score))
    print("save results to " + results_path)

    plt.imshow(haz_des_map, cmap='jet')
    plt.show()

    
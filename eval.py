import cv2
import numpy as np
import scipy.io as scio
import pandas as pd

import model 

if __name__ == "__main__":
    HazDesNet = model.load_HazDesNet()
    HazDesNet.summary()

    y_pred = np.zeros((100, 1))

    data = scio.loadmat('./dataset/LIVE_Defogging/gt.mat')
    y_true = data['subjective_study_mean'].T

    for k in range(100):
        data_file = './dataset/LIVE_Defogging/%d.bmp' % (k + 1)
        img_test = cv2.imread(data_file)

        img_test = np.expand_dims(img_test, axis=0)
        
        y_temp = HazDesNet.predict(img_test)

        y_pred[k, 0] = np.mean(y_temp)

    df = pd.DataFrame({'true': y_true[:, 0], 'pred': y_pred[:, 0]})

    print("SROCC: %.4f" % df.corr('spearman').ix[[0]].values[0][1])
    print("PLCC: %.4f" % df.corr('pearson').ix[[0]].values[0][1])


    
import numpy as np
import cv2

from utils.draw import preprocess
from utils.logger import log


class Extractor(object):
    def __init__(self, model, input_size_h, input_size_w):
        # Return value is with shape (B, D) where D 
        # is the feature dimension.
        self.model = model
        self.feature_array_info = self.model.get_output(0)
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w
        log(f'Init feature extraction {input_size_h}, {input_size_w}')


    def __call__(self, im_crops):
        count = 1
        features = []
        for img in im_crops:
            log(f'Preprocessing feature extraction for person {count} with input size({self.input_size_h}, {self.input_size_w})')
            img = preprocess(img, self.input_size_h, self.input_size_w, False)
            log(f'Preprocessing completed feature extraction for person {count}')
            log('Feature extraction model starting')
            feature_array = np.empty(self.feature_array_info.get_dims(), dtype=self.feature_array_info.get_type())
            # self.model.batch(0, np.expand_dims(img, axis=0))
            self.model.batch(0, img)
            self.model.flush()
            result = self.model.get_result()
            batch_0 = result.get(0)
            batch_0.get(0, feature_array)
            print(feature_array.shape)
            features.append(feature_array[0])
            self.model.release_result(result)
            log('Feature extraction model completed')
            count += 1

        return features



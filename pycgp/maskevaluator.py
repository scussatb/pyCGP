import random

import cv2
import numpy as np
import sys
import os
from roifile import ImagejRoi
import zipfile
import pandas as pd
from skimage import color

from .cgp import CGP
from .evaluator import Evaluator

class MaskEvaluator(Evaluator):
    
    def __init__(self, dirname, dataset_name='dataset.csv', display_dataset=False, resize=1.0, include_hsv=False, include_hed=False, number_of_evaluated_images=-1, subset = 'training'):
        super().__init__()
        self.dirname = dirname
        self.dataset_name = dataset_name
        self.resize = resize
        self.include_hsv = include_hsv
        self.include_hed = include_hed
        self.display_dataset = display_dataset
        self.number_of_evaluated_images = number_of_evaluated_images
        self.subset = subset

        dataset = pd.read_csv(dirname + dataset_name, sep=';')
        print(dataset)
        
        # reading inputs
        self._read_inputs(dataset)
        
        # reading mask
        self._read_mask(dataset)


        if display_dataset:
            for im in range(len(self.input_channels)):
                cv2.imshow('image', self.original_image[im])
                for ch in range(len(self.input_channels[im])):
                    cv2.imshow('channel_'+str(ch), self.input_channels[im][ch])
                    print(self.input_channels[im][ch].shape)
                
                for m in range(len(self.target_masks[im])):
                    cv2.imshow('mask_'+str(m), self.target_masks[im][m])
                    print(self.target_masks[im][m].shape)
                
                cv2.waitKey(0)
                
        self.n_inputs = len(self.input_channels[0])
        self.n_outputs = len(self.target_masks[0])
        
        
    def _read_inputs(self, dataset):
        input_filenames = dataset[dataset['set']==self.subset]['input']
        input_filenames = np.array(input_filenames)
        self.in_image_origin_size = []
        self.original_image = []
        self.input_channels = []
        print(input_filenames)
        for i in range(len(input_filenames)):
            filename = self.dirname + input_filenames[i]

            in_image_origin = cv2.imread(filename)
            self.in_image_origin_size.append(in_image_origin.shape)

            if (self.include_hsv):
                in_image_origin_hsv = cv2.cvtColor(in_image_origin.copy(), cv2.COLOR_BGR2HSV)
                in_image_origin_hsv = cv2.resize(in_image_origin_hsv, (0,0), fx=self.resize, fy=self.resize)

            if (self.include_hed):
                img_rgb = cv2.cvtColor(in_image_origin, cv2.COLOR_BGR2RGB)
                img_hed = color.rgb2hed(img_rgb)
                for i in range(3):
                    img_hed[:, :, i] = img_hed[:, :, i] / img_hed[:, :, i].max()
                in_image_origin_hed = (img_hed * 255).astype(np.uint8)
                in_image_origin_hed = cv2.resize(in_image_origin_hed, (0,0), fx=self.resize, fy=self.resize)

            in_image_origin = cv2.resize(in_image_origin, (0,0), fx=self.resize, fy=self.resize)
            self.original_image.append(in_image_origin)
            
            new_image_in = [in_image_origin[:,:,i] for i in range(in_image_origin.shape[2])]
            if self.include_hsv:
                new_image_in.append(in_image_origin_hsv[:,:,0])
                new_image_in.append(in_image_origin_hsv[:,:,1])
                new_image_in.append(in_image_origin_hsv[:,:,2])
            if self.include_hed:
                new_image_in.append(in_image_origin_hed[:,:,0])
                new_image_in.append(in_image_origin_hed[:,:,1])
                new_image_in.append(in_image_origin_hed[:,:,2])

            self.input_channels.append(new_image_in)
            
                
    def _read_mask(self, dataset):
        self.target_masks = []
        print(dataset.shape)
        print(len(self.in_image_origin_size))

        subdataset = dataset[dataset['set']==self.subset]
        for i_im in range(subdataset.shape[0]):
            im_masks = []
            for i_col in range(2, len(subdataset.columns)):
                roi_filename = self.dirname + subdataset.iloc[i_im][i_col]
                mask = np.zeros(self.in_image_origin_size[i_im])
                if roi_filename[-4:]=='.roi':
                    polygons = self.read_polygons_from_roi(roi_filename)
                    self.fill_polygons_as_labels(mask, polygons)
                elif roi_filename[-4:]=='.zip':
                    zip = zipfile.ZipFile(roi_filename)
                    zip.extractall(roi_filename[:-4])
                    for f in os.listdir(roi_filename[:-4]):
                        polygons = self.read_polygons_from_roi(roi_filename[:-4]+'/'+f)
                        self.fill_polygons_as_labels(mask, polygons)
                elif roi_filename[-4:]=='.png':
                    mask = cv2.imread(roi_filename)

                if mask.max() <= 1.0:
                    mask = mask * 255
                mask = mask.astype(np.uint8)
                mask = cv2.resize(mask, (0,0), fx=self.resize, fy=self.resize)[:,:,0]
                im_masks.append(mask)
            self.target_masks.append(im_masks)

    def is_cacheable(self, it):
        return self.number_of_evaluated_images <= 0

    def evaluate(self, cgp, it, displayTrace = False):
        fit = 0
        vals_iou = []
        if displayTrace:
            print (cgp.to_function_string(['ch_'+str(i) for i in range(self.n_inputs)], ['mask_'+str(i) for i in range(self.n_outputs)]))
#            print('final fitness: ' + str(fit))

        if self.number_of_evaluated_images <= 0:
            images_to_evaluate = range(len(self.input_channels))
        else:
            images_to_evaluate = np.zeros(self.number_of_evaluated_images, dtype=int)
            for i in range(self.number_of_evaluated_images):
                images_to_evaluate[i] = random.randint(0, len(self.input_channels)-1)

        for i_im in images_to_evaluate:
            if displayTrace:
                for i_ch in range(len(self.input_channels[i_im])):
                    cv2.imshow('channel'+str(i_ch), self.input_channels[i_im][i_ch])

            cgp_masks = cgp.run(self.input_channels[i_im].copy()).copy() # !!!!


            for i_mask in range(self.n_outputs):

                cgp_mask = cgp_masks[i_mask]
                val_iou = self.compute_iou(self.target_masks[i_im][i_mask], cgp_mask.copy())
                vals_iou.append(val_iou)
            
                if displayTrace:

                    print(val_iou)
                    y_pred = cgp_mask.copy()
                    y_pred[y_pred > 0] = 255
                    cv2.imshow('target', self.target_masks[i_im][i_mask])
                    cv2.imshow('cgp', y_pred)
                    cv2.waitKey(0)

        # Average weighed by the worsts
        vals_iou.sort(reverse = True)
        fit = 0
        div = 0
        for i in range(len(vals_iou)):
            fit += vals_iou[i] * (i+1) * (i+1)
            div += (i+1)*(i+1)
        fit = fit / div

        # Worth
        #fit = np.min(vals_iou)
        if displayTrace:
#            print (cgp.to_function_string(['ch_'+str(i) for i in range(self.n_inputs)], ['mask_'+str(i) for i in range(self.n_outputs)]))
            print('final fitness: ' + str(fit))


        return fit

    def clone(self):
        return MaskEvaluator(dirname=self.dirname,
                             dataset_name=self.dataset_name,
                             display_dataset=self.display_dataset,
                             resize=self.resize,
                             include_hsv=self.include_hsv,
                             include_hed=self.include_hed,
                             number_of_evaluated_images=self.number_of_evaluated_images,
                             subset=self.subset)

    def compute_iou(self, y_true, y_pred):
        # MetricIOU
        _y_true = y_true
        _y_pred = y_pred
        _y_pred[_y_pred > 0] = 1
        if np.sum(_y_true) == 0:
            _y_true = 1 - _y_true
            _y_pred = 1 - _y_pred
        intersection = np.logical_and(_y_true, _y_pred)
        union = np.logical_or(_y_true, _y_pred)
        return np.sum(intersection) / np.sum(union)
        
    def read_polygons_from_roi(self, filename, scale=1.0):
        print(filename)
        rois = ImagejRoi.fromfile(filename)
        print(rois)
        if type(rois) == ImagejRoi:
            return [rois.coordinates()]
        polygons = [roi.coordinates() for roi in rois]
        return polygons

    def fill_polygons_as_labels(self, mask, polygons):
        for i, polygon in enumerate(polygons):
            cv2.fillPoly(mask, pts=np.int32([polygon]), color=i + 1)
        return mask

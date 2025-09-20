#!/usr/bin/env python3

intervention = 'Smiling'
post_intervention = ['Attractive','Blurry','Mouth_Slightly_Open','High_Cheekbones','Oval_Face','Rosy_Cheeks','Arched_Eyebrows','Bags_Under_Eyes'] # columns to drop because they (potentially) occur after the intervention but before the image
k = 200 # k for knn

#######################################
# number of cpus to use
from multiprocessing import cpu_count
num_cpus = max(cpu_count()-2,1)


#######################################
# lightgbm helper function
import optuna.integration.lightgbm as lgb
from optuna import logging

def fit_lightgbm_model(X_train, 
                       y_train, 
                       X_val, 
                       y_val,
                       objective="regression", 
                       metric="rmse",
                       custom_obj=None,
                       custom_eval=None,
                       weights_train=None,
                       weights_val=None,
                       alpha=None):
    """
    Fit a LightGBM model to the given data, optionally using a custom objective function and a custom evaluation function.

    :param X_train: Feature training data.
    :type X_train: pandas.DataFrame
    :param y_train: Target training data.
    :type y_train: pandas.Series or numpy.ndarray
    :param X_val: Feature validation data.
    :type X_val: pandas.DataFrame
    :param y_val: Target validation data.
    :type y_val: pandas.Series or numpy.ndarray
    :param objective: Objective for LightGBM model, default is "regression".
    :type objective: str, optional
    :param metric: Metric for LightGBM model, default is "rmse".
    :type metric: str, optional
    :param custom_obj: Optional custom objective function for training.
    :type custom_obj: callable, optional
    :param custom_eval: Optional custom evaluation function for validation.
    :type custom_eval: callable, optional
    :param weights_train: Optional instance weights for training data.
    :type weights_train: pandas.Series or numpy.ndarray, optional
    :param weights_val: Optional instance weights for validation data.
    :type weights_val: pandas.Series or numpy.ndarray, optional
    :param alpha: Optional alpha parameter for the objective function.
    :type alpha: float, optional
    :return: Trained LightGBM Booster object.
    :rtype: lightgbm.Booster
    """
    # Define base parameters
    params = {
        "objective": objective,
        "alpha": alpha,
        "metric": metric,
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    logging.set_verbosity(logging.WARNING)

    # Create datasets for LightGBM, incorporating any weights if provided
    train_data = lgb.Dataset(X_train, y_train, weight=weights_train)
    val_data = lgb.Dataset(X_val, y_val, weight=weights_val)

    # Instantiate tuner and perform tuning
    tuner = lgb.LightGBMTuner(params, train_data,
                              valid_sets=[val_data],
                              fobj=custom_obj, feval=custom_eval,
                              show_progress_bar=False,
                              callbacks=[lgb.early_stopping(10, verbose=False)])

    # Run the tuning
    tuner.run()

    # Use the best model
    fit = tuner.get_best_booster()

    return fit



#######################################
# function to estimate the nuisance functions
from datasets import load_dataset, load_from_disk
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import faiss
# Auto-detect and use all available CPUs
faiss.omp_set_num_threads(num_cpus)
import numpy as np
np.random.seed(42)  # for reproducibility

def estimate_nuisances(dataset,k):

    def ip_train_objective(preds, train_data):
        S = train_data.get_label()
        return np.where(S == 1, 2*preds-2, -2), np.where(S == 1, 2, 0) # gradient, hessian
    
    def ip_eval_objective(preds, eval_data):
            S = eval_data.get_label()
            return 'IntSqErr', np.mean(np.where(S == 1, preds**2-2*preds, -2*preds)), False # eval_name, eval_result, is_higher_better
    
    
    df = dataset.to_pandas()
    df[intervention] = (df[intervention] + 1) / 2
    df['no_wgt'] = df[intervention] / df[intervention].mean()
    for ipw_spec in ['right','wrong']:
        df[f'ipw_{ipw_spec}'] = np.nan # will be filled in below
        for outcome_spec in ['right','wrong']:
            df[f'aipw_{ipw_spec}Prop_{outcome_spec}Out'] = np.nan # will be filled in below
            if ipw_spec=='right':
                df[f'gcomp_{outcome_spec}'] = np.nan # will be filled in below
    
    # Split the data in two
    indices = np.arange(len(df))
    np.random.shuffle(indices)
    
    # Split indices into two parts
    split_point = len(indices) // 2
    index_sets = [
        {'nuis_fit': indices[:split_point], 'nuis_eval': indices[split_point:]},
        {'nuis_fit': indices[split_point:], 'nuis_eval': indices[:split_point]}
    ]

    # First nuisance: inverse propensity
    for ipw_spec in ['right','wrong']: # is the propensity score model correct or incorrect?
        for inds in index_sets:
            df_train_ipw = df.iloc[inds['nuis_fit']]
                
            X_train_ipw = df_train_ipw.drop([intervention,'image','ipw_right','ipw_wrong','gcomp_right','gcomp_wrong','aipw_rightProp_rightOut','aipw_wrongProp_rightOut','aipw_rightProp_wrongOut','aipw_wrongProp_wrongOut'], axis=1)
            y_train_ipw = df_train_ipw[intervention]
            X_val_ipw = df.drop([intervention,'image','ipw_right','ipw_wrong','gcomp_right','gcomp_wrong','aipw_rightProp_rightOut','aipw_wrongProp_rightOut','aipw_rightProp_wrongOut','aipw_wrongProp_wrongOut'], axis=1).iloc[inds['nuis_eval']]
            y_val_ipw = df[intervention].iloc[inds['nuis_eval']]                
    
            my_fit = fit_lightgbm_model(
                X_train_ipw, 
                y_train_ipw, 
                X_val_ipw, 
                y_val_ipw,
                objective="custom",
                metric='IntSqErr',
                custom_obj=ip_train_objective,
                custom_eval=ip_eval_objective)
            # clip inverse propensity at 1 (since 1/probability >= 1)
            unscaled = y_val_ipw * np.maximum(my_fit.predict(X_val_ipw),1)
            # if wrong, downweight non-black-haired people by 4
            unscaled = unscaled * (1 - .75*((1-df['Black_Hair'].iloc[inds['nuis_eval']])/2 if ipw_spec=='wrong' else 0))
            # stabilize ip weights so they have mean 1
            df.loc[inds['nuis_eval'], f'ipw_{ipw_spec}'] = unscaled/unscaled.mean()

    # Second nuisance: conditional distribution of image given features
    for outcome_spec in ['right','wrong']: # is the conditional outcome generative model correct or incorrect?
        reverse_matched_inds = [[] for _ in range(len(df))]
        for inds in index_sets:
            X_train = df.drop([intervention,'image','ipw_right','ipw_wrong','gcomp_right','gcomp_wrong','aipw_rightProp_rightOut','aipw_wrongProp_rightOut','aipw_rightProp_wrongOut','aipw_wrongProp_wrongOut'], axis=1).iloc[inds['nuis_fit']]
            y_train = df[intervention].iloc[inds['nuis_fit']]
            X_val = df.drop([intervention,'image','ipw_right','ipw_wrong','gcomp_right','gcomp_wrong','aipw_rightProp_rightOut','aipw_wrongProp_rightOut','aipw_rightProp_wrongOut','aipw_wrongProp_wrongOut'], axis=1).iloc[inds['nuis_eval']]
            y_val = df[intervention].iloc[inds['nuis_eval']]                
    
            # estimate via knn
            if outcome_spec=='right':
                treatment_mask = df[intervention].iloc[inds['nuis_fit']] == 1
            else:
                treatment_mask = (df[intervention].iloc[inds['nuis_fit']] == 1) & (df['Black_Hair'].iloc[inds['nuis_fit']] == 1)

            # X_train_filtered = X_train[treatment_mask]
            X_train_filtered = (X_train.drop(['Black_Hair'], axis=1) if outcome_spec=='wrong' else X_train).loc[treatment_mask.index[treatment_mask]]
            treated_indices = np.array(inds['nuis_fit'])[treatment_mask]  # Keep track of original indices
            
            X_train_array = np.ascontiguousarray(X_train_filtered.to_numpy(), dtype='float32')
            X_val_array = np.ascontiguousarray((X_val.drop(['Black_Hair'], axis=1) if outcome_spec=='wrong' else X_val).to_numpy(), dtype='float32')
            
            d = X_train_array.shape[1]
            nlist = int(np.sqrt(len(X_train_array)))
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
            
            index.train(X_train_array)
            index.add(X_train_array)
            index.nprobe = min(nlist, 10)
            
            _, I = index.search(X_val_array, k=k)
            
            matched_treated = treated_indices[I]
            for i in range(len(I)):
                eval_idx = inds['nuis_eval'][i]
                for matched_idx in matched_treated[i]:
                    reverse_matched_inds[matched_idx].append(eval_idx)
                    
        for ipw_spec in ['right','wrong']: # is the propensity score model correct or incorrect?
            for i in range(len(df)):
                if reverse_matched_inds[i]:  # Check if there are any reverse matches
                    df.loc[i, f'aipw_{ipw_spec}Prop_{outcome_spec}Out'] = df.loc[i, f'ipw_{ipw_spec}'] + (1 - df.loc[reverse_matched_inds[i], f'ipw_{ipw_spec}']).sum()/k
                    if ipw_spec=='right':
                        df.loc[i, f'gcomp_{outcome_spec}'] = (1 - 0*df.loc[reverse_matched_inds[i], f'ipw_{ipw_spec}']).sum()/k
                else:
                    df.loc[i, f'aipw_{ipw_spec}Prop_{outcome_spec}Out'] = df.loc[i, f'ipw_{ipw_spec}']
                    if ipw_spec=='right':
                        df.loc[i, f'gcomp_{outcome_spec}'] = 0
                        

    return df['ipw_right'].to_numpy(), df['ipw_wrong'].to_numpy(), df['gcomp_right'].to_numpy(), df['gcomp_wrong'].to_numpy(), df['aipw_rightProp_rightOut'].to_numpy(), df['aipw_wrongProp_rightOut'].to_numpy(), df['aipw_rightProp_wrongOut'].to_numpy(), df['aipw_wrongProp_wrongOut'].to_numpy(), df['no_wgt'].to_numpy()
    
#######################################
# face segmentation

import mediapipe as mp
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
from torchvision import transforms

def process_batch(examples, worker_id=None, **kwargs):
    """
    Hybrid approach combining MediaPipe and SAM for face segmentation.
    Returns RGBA images with the segmentation mask as alpha channel.
    """
    checkpoint_path = kwargs.get('checkpoint_path', 'sam_vit_h_4b8939.pth')
    threshold = kwargs.get('threshold', 0.66)
    blur_radius = kwargs.get('blur_radius', 3)
    padding = kwargs.get('padding', 0.25)
    morph_kernel_size = kwargs.get('morph_kernel_size', 2)
    morph_iterations = kwargs.get('morph_iterations', 2)
    min_area_fraction = kwargs.get('min_area_fraction', 0.2)

    # Initialize MediaPipe
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    selfie_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Initialize SAM
    gpu_id = (worker_id or 0) % torch.cuda.device_count()
    device = torch.device(f'cuda:{gpu_id}')
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    def create_mediapipe_mask(image_np, face_results):
        """Create improved MediaPipe mask combining face detection and selfie segmentation"""
        h, w = image_np.shape[:2]
        detection_mask = np.zeros((h, w), dtype=np.uint8)
        
        if face_results.detections:
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Add padding for face region
            pad_x = int(width * padding)
            pad_y = int(height * padding)
            
            # Calculate ellipse parameters
            center_x = x + width // 2
            center_y = y + height // 2
            axis_x = (width + 2 * pad_x) // 2
            axis_y = (height + 2 * pad_y) // 2
            
            # Create gradient ellipse mask
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = ((x_coords - center_x)**2 / (axis_x**2) + 
                              (y_coords - center_y)**2 / (axis_y**2))
            gradient_mask = np.clip(1.5 - dist_from_center, 0, 1)
            detection_mask = (gradient_mask * 255).astype(np.uint8)
        
        # Get selfie segmentation mask
        selfie_results = selfie_segmenter.process(image_np)
        selfie_mask = (selfie_results.segmentation_mask > threshold)
        selfie_mask = selfie_mask.astype(np.uint8) * 255
        
        # Blend masks using multiplication (logical AND with scaling)
        combined_mask = cv2.multiply(detection_mask, selfie_mask, scale=1/255)
        
        # Enhance hair regions from selfie mask
        hair_regions = cv2.subtract(selfie_mask, combined_mask)
        hair_regions = cv2.GaussianBlur(hair_regions, (blur_radius, blur_radius), 0)
        
        # Add hair regions back to combined mask
        combined_mask = cv2.add(combined_mask, hair_regions)
        
        return combined_mask

    def get_face_bbox(image_np):
        """Get face bounding box from MediaPipe"""
        face_results = face_detector.process(image_np)
        if not face_results.detections:
            return None, face_results
        
        h, w = image_np.shape[:2]
        detection = face_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Convert relative coordinates to absolute with padding
        x = int(max(0, bbox.xmin * w - padding * w))
        y = int(max(0, bbox.ymin * h - padding * h))
        width = int(min(w - x, bbox.width * w + 2 * padding * w))
        height = int(min(h - y, bbox.height * h + 2 * padding * h))
        
        return (x, y, width, height), face_results

    def get_sam_mask(image_np, bbox):
        """Get segmentation mask from SAM using face bbox as prompt"""
        predictor.set_image(image_np)
        
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        
        points = np.array([
            [center_x, center_y],  # Face center
            [x + w//4, y + h//4],  # Upper left
            [x + 3*w//4, y + h//4],  # Upper right
            [x + w//2, y + h//6],  # Forehead
            [x + w//2, y + 5*h//6]  # Chin
        ])
        labels = np.ones(len(points))  # All points are foreground
        
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )
        
        # Select the mask with the highest score
        best_mask = masks[np.argmax(scores)]
        return best_mask.astype(np.uint8) * 255

    def combine_masks(mp_mask, sam_mask):
        """Combine MediaPipe and SAM masks"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mp_mask = cv2.morphologyEx(mp_mask, cv2.MORPH_CLOSE, kernel)
        sam_mask = cv2.morphologyEx(sam_mask, cv2.MORPH_CLOSE, kernel)
        
        combined = cv2.max(mp_mask, sam_mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
        
        num_labels, labels = cv2.connectedComponents(combined)
        min_area = min_area_fraction * (combined.shape[0] * combined.shape[1])
        for label in range(1, num_labels):
            if np.sum(labels == label) < min_area:
                combined[labels == label] = 0
        
        h, w = combined.shape[:2]
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        
        _, binary = cv2.threshold(combined, 127, 255, cv2.THRESH_BINARY)
        flood_image = cv2.bitwise_not(binary)
        
        for x in range(w):
            if flood_image[0, x] == 255:
                cv2.floodFill(flood_image, flood_mask, (x, 0), 0)
            if flood_image[h-1, x] == 255:
                cv2.floodFill(flood_image, flood_mask, (x, h-1), 0)
        for y in range(h):
            if flood_image[y, 0] == 255:
                cv2.floodFill(flood_image, flood_mask, (0, y), 0)
            if flood_image[y, w-1] == 255:
                cv2.floodFill(flood_image, flood_mask, (w-1, y), 0)
        
        holes = (flood_image == 255)
        combined[holes] = 255
        
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.dilate(combined, dilation_kernel, iterations=1)
               
        return combined

    processed_images = []

    for image in examples['image']:
        image_np = np.array(image)
        
        # Get face bbox and results
        bbox, face_results = get_face_bbox(image_np)
        
        if bbox is not None:
            # Regular SAM + MediaPipe approach
            selfie_results = selfie_segmenter.process(image_np)
            mp_mask = (selfie_results.segmentation_mask > threshold).astype(np.uint8) * 255
            sam_mask = get_sam_mask(image_np, bbox)
            final_mask = combine_masks(mp_mask, sam_mask)
        else:
            # MediaPipe fallback
            final_mask = create_mediapipe_mask(image_np, face_results)

        # Double pass on edge smoothing
        final_mask = cv2.GaussianBlur(final_mask, (blur_radius, blur_radius), 1)
        final_mask = cv2.GaussianBlur(final_mask, (blur_radius, blur_radius), 0.5)
        
        # Create RGBA PIL Image directly from RGB image and alpha mask
        final_image = Image.fromarray(image_np, mode='RGB')
        mask_image = Image.fromarray(final_mask, mode='L')
        final_image.putalpha(mask_image)
        
        processed_images.append(final_image)

    face_detector.close()
    selfie_segmenter.close()
    
    return {
        'image': processed_images
    }

#######################################
# cropping artifacts ('smear' that appears on near borders of some of the images)

import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Union

def detect_and_crop_artifacts(examples: Dict[str, List[Union[np.ndarray, torch.Tensor]]], **kwargs) -> Dict[str, List[Image.Image]]:
    """
    Process RGBA arrays to detect and crop artifacts in the top portion.
    Supports debug visualization in Jupyter notebooks.
    """
    processed_images = []
    batch_idx = kwargs.get('batch_idx', 0)

    for idx, img_array in enumerate(examples['image']):
        # Convert to numpy array if needed
        if isinstance(img_array, torch.Tensor):
            img_array = img_array.numpy()
        elif isinstance(img_array, Image.Image):
            img_array = np.array(img_array)  # Handle case where input is PIL Image
            
        # Extract RGB for processing
        rgb_array = img_array[..., :3]
            
        # Convert to grayscale for template matching
        gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
            
        # Get dimensions
        height, width = gray.shape
        top_third_height = height // 3
        
        # Get the middle 70% of the first few rows
        start_x = int(width * 0.15)  # 15% from left
        end_x = int(width * 0.85)    # 15% from right
        template = gray[0:5, start_x:end_x]
        
        # Check for pattern variation in template
        template_std = np.std(template)
        min_variation = 10  # Minimum standard deviation to consider a pattern interesting
        
        needs_cropping = False
        crop_point = 0
        
        if template_std > min_variation:
            # Search in top third of image
            search_region = gray[:top_third_height]
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations where the match is strong
            threshold = 0.8
            locations = np.where(result > threshold)
            y_matches, x_matches = locations
            
            # Process matches if we have enough
            if len(y_matches) >= 3:
                needs_cropping = True
                crop_point = np.max(y_matches) + 2  # Add 2 to account for template height
        
        # Debug visualization (using RGB part)
        if kwargs.get('debug', False):
            global_idx = batch_idx * len(examples['image']) + idx
            print(f'Processing image {global_idx}')
            
            if needs_cropping:
                print(f"Found {len(y_matches)} matches above threshold {threshold}")
                print(f"Template variation (std): {template_std:.2f}")
                print(f"Cropped at y={crop_point}")
                
                # Create figure with all visualizations in one row
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original image
                ax1.imshow(rgb_array)
                ax1.set_title('Original Image')
                
                # Show the template
                ax2.imshow(template, cmap='gray')
                ax2.set_title(f'Template (std={template_std:.2f})')
                
                # Original with crop line and matches
                ax3.imshow(rgb_array)
                ax3.axhline(y=crop_point, color='r', linestyle='-', linewidth=2)
                for y in y_matches:
                    ax3.axhline(y=y, color='yellow', alpha=0.3)
                ax3.set_title('Matches (yellow) and crop line (red)')
                
                # Cropped version
                cropped_array = rgb_array[crop_point:]
                ax4.imshow(cropped_array)
                ax4.set_title('After cropping')
                
                plt.tight_layout()
                plt.show()
            else:
                if template_std <= min_variation:
                    print(f"Template has insufficient variation (std={template_std:.2f}) - skipping")
                else:
                    print(f"Not enough matches found - no cropping needed (found {len(y_matches)} matches)")
                    
                # Show only original image
                plt.figure(figsize=(5, 5))
                plt.imshow(rgb_array)
                plt.title('Original Image')
                plt.show()
        
        # Process the image
        if needs_cropping:
            img_array = img_array[crop_point:]
            
        processed_images.append(img_array)
       
    return {'image': processed_images}

#######################################
# split RGBA into image and mask

def split_rgba_images(examples: Dict[str, List[Image.Image]]) -> Dict[str, List[Image.Image]]:
    """
    Split RGBA PIL Images into three columns:
    - image: RGB PIL Image
    - mask: Grayscale PIL Image (from alpha channel)
    - image_and_mask: RGBA PIL Image (original)
    """
    processed = {
        'image': [],
        'mask': [],
        'image_and_mask': []
    }
    
    for rgba_image in examples['image']:
        # Split the RGBA PIL Image into RGB and Alpha
        rgb_image = rgba_image.convert('RGB')
        # Get alpha channel (masks are stored in channel 'A' of RGBA image)
        mask_image = rgba_image.getchannel('A')
        
        processed['image'].append(rgb_image)
        processed['mask'].append(mask_image)
        processed['image_and_mask'].append(rgba_image)
    
    return processed

#######################################
# process images


dataset = load_dataset("tpremoli/CelebA-attrs", split="train")
dataset = dataset.shuffle(seed=42) # shuffle the rows of the dataset
dataset = dataset.remove_columns(['prompt_string']+post_intervention)

# for testing
# dataset = dataset.select(range(1024))

##############################
# remove backgrounds from images, using mediapipe library
# Verify GPU count

num_gpus = torch.cuda.device_count()
print(f"Available GPUs: {num_gpus}")

# Use one process per GPU
dataset = dataset.map(
    process_batch,
    fn_kwargs={"checkpoint_path": "sam_vit_h_4b8939.pth"},
    batched=True,
    batch_size=min(4096, int(np.ceil(len(dataset)/num_gpus))),
    num_proc=num_gpus,
    with_rank=True,
    desc="Finding foregrounds"
)

##############################
# some of the images have a weird artifact above them. Like the top row of pixels was taken and dragged upwards at an angle. This code aims to crop that part out of those images, and then after that it square crops them
dataset = dataset.map(
    detect_and_crop_artifacts,
    batched=True,
    batch_size=min(222, int(np.ceil(len(dataset)/num_cpus))),
    num_proc=num_cpus,
    fn_kwargs={'debug': False},  # Set to True to see the detection process
    desc="Detecting and cropping artifacts"
)

##############################
# estimate (augmented) inverse probability weights
ipw_right, ipw_wrong, gcomp_right, gcomp_wrong, aipw_rightProp_rightOut, aipw_wrongProp_rightOut, aipw_rightProp_wrongOut, aipw_wrongProp_wrongOut, no_wgt = estimate_nuisances(dataset,k)
dataset = dataset.add_column(name="ipw_right", column=ipw_right)
dataset = dataset.add_column(name="ipw_wrong", column=ipw_wrong)
dataset = dataset.add_column(name="gcomp_right", column=gcomp_right)
dataset = dataset.add_column(name="gcomp_wrong", column=gcomp_wrong)
dataset = dataset.add_column(name="aipw_rightProp_rightOut", column=aipw_rightProp_rightOut)
dataset = dataset.add_column(name="aipw_wrongProp_rightOut", column=aipw_wrongProp_rightOut)
dataset = dataset.add_column(name="aipw_rightProp_wrongOut", column=aipw_rightProp_wrongOut)
dataset = dataset.add_column(name="aipw_wrongProp_wrongOut", column=aipw_wrongProp_wrongOut)
dataset = dataset.add_column(name="no_wgt", column=no_wgt)

##############################
# Convert images. Save 3 copies: image only (RGB), mask only (greyscale), and image_and_mask' (RGBA, with A the mask)
dataset = dataset.map(
    split_rgba_images,
    batched=True,
    batch_size=min(222, int(np.ceil(len(dataset)/num_cpus))),
    num_proc=num_cpus,
    desc="Converting images"
)

######
# Reorder columns
current_columns = dataset.column_names
# Start with image and matched images
new_order = ['image','mask','image_and_mask','ipw_right','ipw_wrong','gcomp_right','gcomp_wrong','aipw_rightProp_rightOut','aipw_wrongProp_rightOut','aipw_rightProp_wrongOut','aipw_wrongProp_wrongOut','no_wgt',intervention]

# Add all other columns except the ones we've already included
other_columns = [col for col in current_columns if col not in new_order]
new_order.extend(other_columns)

# Reorder the dataset columns
dataset = dataset.select_columns(new_order)
######

dataset.save_to_disk("./CelebA-attrs")

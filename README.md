# Cerebral Microbleeds (CMB) segmentation with a 3D Unet

This repository contains the trained tensorflow models for the 3D Segmentation of Cerebral Microbleeds on SWI MR Images with a 3D U-Shaped Neural Network (U-net) as described in a forthcoming scientific publication.

![Gif Image](https://github.com/pboutinaud/SHIVA_PVS/blob/main/docs/Images/SHIVA_BrainTools_small2.gif)

## IP, Licencing & Usage

**The inferences created by these models should not be used for clinical purposes.**

The segmentation models in this repository have been registered at the french 'Association de Protection des Programmes' under the number: 
[IDDN.FR.001.240015.000.S.P.2022.000.31230](https://secure.app.asso.fr/app.server/certificate/?sn=2023420007000&key=b6111d4ba322d83ad2a19f8c09b83da5c23ce23c873a5a99fd9e2892be635da1&lang=fr). 

The segmentation models in this repository are provided under the Creative Common Licence [BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/).

![Creative Common Licence BY-NC-SA](./docs/logos/by-nc-sa.eu_.png)

## The segmentation models
The models were trained with SWI images with a size of 160 × 214 × 176 x 1 voxels. The training was done with images with an isotropic voxel size of 1 × 1 × 1 mm3 and with normalized voxel values in [0, 1] (min-max normalization with the max set to the 99th percentile of the brain voxel values to avoid "hot spots"). The brain is supposed to be centered, the models are trained with and without a brain mask applied on images. This model can also be used with T2*Gre images.

The segmentation can be computed as the average of the inference of several models (depending on the number of folds used in the training for a particular model). The resulting segmentation is an image with voxels values in [0, 1] (proxy for the probability of detection of CMB) that must be thresholded to get the actual segmentation. A threshold of 0.5 has been used successfully but that depends on the preferred balance between precision and sensitivity.

To access the models :
* **v2/SWI-CMB (recommended)**: New production models based on the ResUnet3D architecture, trained with Keras 3 / TensorFlow ≥ 2.17. The models are stored in the TensorFlow SavedModel format (3 folds).
    * Download: [cloud.efixia.com](https://cloud.efixia.com/sharing/hwR97APpm)
    * SHA256 checksum : DC4E3A8E92C32657A172754B0971E849737157AF07D2A229E58016D0F3A4229B
    * JSON file for SHiVAi pipeline: [model_info_swi-cmb-v2.json](model_info_swi-cmb-v2.json)

* v1/SWI-CMB: The segmentation models described in the publication, stored in H5 format (3 folds). Requires TensorFlow ≥ 2.7 with legacy Keras (tf-keras).
    * due to file size limitation the models can be found [here](https://cloud.efixia.com/sharing/mb4gM77BK) https://cloud.efixia.com/sharing/mb4gM77BK
    * MD5 checksum : 998ab737afc83ae48a193803c50cf3de
    * JSON file for SHiVAi pipeline: [model_info_swi-cmb-v1.json](model_info_swi-cmb-v1.json)

## Requirements

### For new models (v2, SavedModel format)
The models require TensorFlow ≥ 2.17 and were tested with Python 3.12 and TensorFlow 2.20. They are stored in the TensorFlow SavedModel format. A NVIDIA GPU with at least 9 GB of VRAM is recommended for inference (CPU inference is also supported but slower).

### For legacy models (v1, H5 format)
The models were trained with TensorFlow ≥ 2.7 and Python 3.7, stored in H5 format. Loading these models with newer Python/TensorFlow versions requires the `tf-keras` compatibility package and the environment variable `TF_USE_LEGACY_KERAS=1` set before importing TensorFlow. On CPU, models using mixed_float16 are automatically rebuilt in float32.

### Python dependencies
To run the `predict_one_file.py` script, you will need a python environment with the following libraries:
- tensorflow >= 2.17 (for new models) or tensorflow >= 2.7 (for legacy models)
- numpy
- nibabel
- tf-keras (only needed for legacy .h5 models)

If you don't know anything about python environment and libraries, you can find some documentation and installers on the [Anaconda website](https://docs.anaconda.com/). We recommend using the lightweight [Miniconda](https://docs.anaconda.com/miniconda/).

## Usage
**These models can be used with the [SHiVAi](https://github.com/pboutinaud/SHiVAi) preprocessing and deep learning segmentation workflow.**

### Step-by-step process to run the model without SHiVAi
1. Download the `predict_one_file.py` from the repository (clic the "<> Code" button on the GitHub interface and download the zip file, or clone the repository)
2. Download and unzip the trained models (see [above](#the-segmentation-models))
3. Preprocess the input data (swi or T2gre images) to the proper x-y-z volume (160 × 214 × 176). If the resolution is close to 1mm isotropic voxels, a simple cropping is enough. Otherwise, you will have to resample the images to 1mm isotropic voxels. For now, you will have to do it by yourself, but soon we will provide a full Shiva pipeline to run everything.
4. Run the `predict_one_file.py` script as described below

To run `predict_one_file.py` in your python environment you can check the help with the command `python predict_one_file.py -h` (replace "predict_one_file.py" with the full path to the script if it is not in the working directory).

Here is an example of usage of the script with the new SavedModel models:
- The `predict_one_file.py` script stored in `/myhome/my_scripts/`
- Preprocessed Nifti images (volume shape must be 160 × 214 × 176 and voxel values between 0 and 1) stored (for the example) in the folder `/myhome/mydata/`
- The CMB AI models stored (for the example) in `/myhome/cmb_models/v2`
- The ouput folder (for the example) `/myhome/my_results` needs to exist at launch

```bash
# New SavedModel models (v2, recommended)
python /myhome/my_scripts/predict_one_file.py \
    -i /myhome/mydata/swi_image.nii.gz \
    -b /myhome/mydata/input_brainmask.nii.gz \
    -o /myhome/my_results/cmb_segmentation.nii.gz \
    --batch_size 1 --gpu 0 \
    -m /myhome/cmb_models/v2/20250129-192041_ResUnet3D-8.9.2-1.5-SWAN.CMB_prod2_fold_0_bestvalloss.tf_inference \
    -m /myhome/cmb_models/v2/20250129-195344_ResUnet3D-8.9.2-1.5-SWAN.CMB_prod2_fold_1_bestvalloss.tf_inference \
    -m /myhome/cmb_models/v2/20250129-192041_ResUnet3D-8.9.2-1.5-SWAN.CMB_prod2_fold_2_bestvalloss.tf_inference
```
>Note that the brain mask input here with `-b /myhome/mydata/input_brainmask.nii.gz` is optional

```bash
# Legacy H5 models (v1)
python /myhome/my_scripts/predict_one_file.py \
    -i /myhome/mydata/swi_image.nii.gz \
    -b /myhome/mydata/input_brainmask.nii.gz \
    -o /myhome/my_results/cmb_segmentation.nii.gz \
    --gpu 0 \
    -m /myhome/cmb_models/v1/20230330-095525_Unet3Dv2-10.7.2-1.8-SWAN.CMB_fold_CMB_1x3-331_fold_1_model.h5 \
    -m /myhome/cmb_models/v1/20230329-221948_Unet3Dv2-10.7.2-1.8-SWAN.CMB_fold_CMB_1x3-331_fold_2_model.h5 \
    -m /myhome/cmb_models/v1/20230328-101347_Unet3Dv2-10.7.2-1.8-SWAN.CMB_fold_CMB_1x3-331_fold_0_model.h5
```
>Note that the brain mask input here with `-b /myhome/mydata/input_brainmask.nii.gz` is optional

### Building your own script
The provided python script `predict_one_file.py` can be used as is for running the model or can be used an example to build your own script.

Here is the main part of the script for new SavedModel models:
````python
import tensorflow as tf
import numpy as np

# Load models & predict
predictions = []
for model_dir in model_dirs:  # model_dirs is the list of SavedModel directory paths
    model = tf.saved_model.load(model_dir)
    # For SavedModel: use model.serve() with manual batching
    batch = tf.constant(images, dtype=tf.float32)
    prediction = model.serve(batch).numpy()
    predictions.append(prediction)

# Average all predictions
predictions = np.mean(predictions, axis=0)
````

For legacy .h5 models (requires `tf-keras` and `TF_USE_LEGACY_KERAS=1`):
````python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import numpy as np

predictions = []
for predictor_file in predictor_files:
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(predictor_file, compile=False, custom_objects={"tf": tf})
    prediction = model.predict(images)
    predictions.append(prediction)

predictions = np.mean(predictions, axis=0)
````

## Acknowledgements
This work has been done in collaboration between the [Fealinx](http://www.fealinx-biomedical.com/en/) company and the [GIN](https://www.gin.cnrs.fr/en/) laboratory (Groupe d'Imagerie Neurofonctionelle, UMR5293, IMN, Univ. Bordeaux, CEA , CNRS) with grants from the Agence Nationale de la Recherche (ANR) with the projects [GinesisLab](http://www.ginesislab.fr/) (ANR 16-LCV2-0006-01) and [SHIVA](https://rhu-shiva.com/en/) (ANR-18-RHUS-0002)

|<img src="./docs/logos/shiva_blue.png" width="100" height="100" />|<img src="./docs/logos/fealinx.jpg" height="200" />|<img src="./docs/logos/Logo-Gin.png" height="200" />|<img src="./docs/logos/logo_ginesis-1.jpeg" height="100" />|<img src="./docs/logos/logo_anr.png" height="50" />|
|---|---|---|---|---|

## Publication

https://www.nature.com/articles/s41598-024-81870-5

Tsuchida, A., Goubet, M., Boutinaud, P., Astafeva, I., Nozais, V., Hervé, P. Y., Tourdias, T., Debette, S., & Joliot, M. (2024). SHIVA-CMB: a deep-learning-based robust cerebral microbleed segmentation tool trained on multi-source T2*GRE- and susceptibility-weighted MRI. Scientific reports, 14(1), 30901. 

https://doi.org/10.1038/s41598-024-81870-5
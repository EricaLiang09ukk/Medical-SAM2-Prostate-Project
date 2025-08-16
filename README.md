<h1 align="center">● Evaluating MedSAM-2 for Prostate MRI Segmentation</h1>

<p align="center">
    <a href="https://discord.gg/DN4rvk95CC">
        <img alt="Discord" src="https://img.shields.io/discord/1146610656779440188?logo=discord&style=flat&logoColor=white"/></a>
    <img src="https://img.shields.io/static/v1?label=license&message=GPL&color=white&style=flat" alt="License"/>
</p>

Medical SAM 2, or say MedSAM-2, is an advanced segmentation model that utilizes the [SAM 2](https://github.com/facebookresearch/segment-anything-2) framework to address both 2D and 3D medical
image segmentation tasks. This model and methods were adapted in an attempt to use the model for segmentation of the prostate, as well as prostate lesions.

## 🔥 A Quick Overview 
 <div align="center"><img width="880" height="350" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/framework.png"></div>
 
## 🩻 3D Abdomen Segmentation Visualisation
 <div align="center"><img width="420" height="420" src="https://github.com/MedicineToken/Medical-SAM2/blob/main/vis/example.gif"></div>

## 🧐 Requirement

 Install the environment:

 ``conda env create -f environment.yml``

 ``conda activate medsam2``

 You can download SAM2 checkpoint from checkpoints folder:
 
 ``bash download_ckpts.sh``

 Further Note: The model was tested on the following system environment and you may have to handle some issue due to system difference.
```
Operating System: Ubuntu 22.04
Conda Version: 23.7.4
Python Version: 3.12.4
```
Download the MedSAM-2 pretrained weights [here](https://huggingface.co/jiayuanz3/MedSAM2_pretrain/tree/main)

 ## 🎯 Example Cases
 #### Download Prostate-MRI-US-Biopsy or your own dataset and put in the ``data`` folder, create the folder if it does not exist ⚒️

**Step1:** Download the [prostate-MRI-US-Biopsy](https://portal.imaging.datacommons.cancer.gov/explore/filters/?collection_id=prostate_mri_us_biopsy) dataset from the Imaging Data Commons using instructions provided:

**Step2:** Preprocess the data using the Data_preprocessing notebook.

**Step3:** Run the training and validation by:
 
 ``python train_3d.py -exp_name prostate_mri_MedSAM2 -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -pretrain MedSAM2_pretrain.pth -image_size 1024 -dataset prostate_mri -data_path ./data/prostate_mri``


## 📝 Citation
 ~~~
@misc{zhu_medical_2024,
	title={Medical SAM 2: Segment medical images as video via Segment Anything Model 2},
    author={Jiayuan Zhu and Yunli Qi and Junde Wu},
    year = {2024},
    eprint={2408.00874},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
 ~~~

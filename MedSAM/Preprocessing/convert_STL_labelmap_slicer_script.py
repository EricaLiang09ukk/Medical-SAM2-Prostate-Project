# Run this code on 3D slicer

import os
import glob

for stl_file_name in glob.glob(os.path.join("D:/CISC867/PKG - Prostate-MRI-US-Biopsy-STL/STLs/STLs/*ProstateSurface*.STL")):

    stl_file_name = stl_file_name.replace('\\', '/')
    output_file_name = f"D:/CISC867/PKG - Prostate-MRI-US-Biopsy-STL/T2_prostate_mask/{stl_file_name.split('/')[-1][:-4]}.nii.gz"
    output_file_name = os.path.abspath(output_file_name).replace('\\', '/')

    patID = "-".join(stl_file_name.split("/")[-1].split('-')[0:5])
    serieUID = stl_file_name.split("/")[-1].split("-")[-1][:-4]

    patient_folder = os.path.join("D:/CISC867/prostate_mri_us_biopsy", patID).replace('\\', '/')
    nrrd_path = os.path.join("D:/CISC867/PKG - Prostate-MRI-US-Biopsy-STL/T2MR_nrrd", f"{patID}-ProstateSurface-seriesUID-{serieUID}.nrrd")
    nrrd_path = os.path.abspath(nrrd_path).replace('\\', '/')
    
    if os.path.exists(nrrd_path):
        reference_volume_path_lst = nrrd_path

        try:
            referenceVolumeNode = slicer.util.loadVolume(reference_volume_path_lst)
            if referenceVolumeNode is None:
                print(f"Loading reference volume fail: {reference_volume_path_lst}")
                continue

            segmentationNode = slicer.util.loadSegmentation(stl_file_name)
            if segmentationNode is None:
                print(f"Loading STL fail: {stl_file_name}")

            outputLabelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, outputLabelmapVolumeNode, referenceVolumeNode)
            
            slicer.util.saveNode(outputLabelmapVolumeNode, output_file_name)
            
            slicer.mrmlScene.RemoveNode(segmentationNode)
            slicer.mrmlScene.RemoveNode(outputLabelmapVolumeNode)
            slicer.mrmlScene.RemoveNode(referenceVolumeNode)
            slicer.mrmlScene.Clear(0)

        except Exception as e:
            print(f"Error: {stl_file_name} : {e}")


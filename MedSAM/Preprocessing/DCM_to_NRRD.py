# Run this code on 3D slicer

import os
import glob


def convert_dicom_to_nrrd(dicom_folder, output_file_name):
    plugin = slicer.modules.dicomPlugins['DICOMScalarVolumePlugin']()

    fileList = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.endswith('.dcm')]
    if not fileList:
        print(f"Can not find dicom files：{dicom_folder}")
        return False

    loadables = plugin.examine([fileList])
    if not loadables:
        print(f"Unable to load dicom files：{dicom_folder}")
        return False

    volume_node = plugin.load(loadables[0])
    slicer.util.saveNode(volume_node, output_file_name)
    slicer.mrmlScene.RemoveNode(volume_node)
    return True


for stl_file_name in glob.glob(os.path.join("D:/CISC867/PKG - Prostate-MRI-US-Biopsy-STL/STLs/STLs/*ProstateSurface*.STL")):
    stl_file_name = stl_file_name.replace('\\', '/')
    output_file_name = f"D:/CISC867/PKG - Prostate-MRI-US-Biopsy-STL/T2MR_nrrd/{stl_file_name.split('/')[-1][:-4]}.nrrd"

    patID = "-".join(stl_file_name.split("/")[-1].split('-')[0:5])
    serieUID = stl_file_name.split("/")[-1].split("-")[-1][:-4]

    patient_folder = os.path.join("D:/CISC867/prostate_mri_us_biopsy", patID).replace('\\', '/')
    T2MR_dcm_folder = f"MR_{serieUID}"
    print(T2MR_dcm_folder)

    T2MR_dcm_path = None
    for roots, dirs, files in os.walk(patient_folder):
        for dir_name in dirs:
            if dir_name == T2MR_dcm_folder:
                T2MR_dcm_path = os.path.join(roots, dir_name).replace('\\', '/')
                break
        if T2MR_dcm_path:
            break

    if T2MR_dcm_path:
        success = convert_dicom_to_nrrd(T2MR_dcm_path, output_file_name)
        if not success:
            print(f"Convertion fail: {T2MR_dcm_path}")
    else:
        print(f"No dicom file found：{T2MR_dcm_folder} for {stl_file_name}")

print("Done")

import glob
import SimpleITK as sitk
import os

# GADF filter function imported
from gradientAnisotropicDiffusionFilter import gADFilter
from gradientAnisotropicDiffusionFilter import plotGADF
from N4Bias_Field_Correction import n4_bias_field_correction
from registrationF import registration


def main():
    # Function for loading the images from each of the patient folders and splitting them into 3 arrays
    fileListFLAIR = glob.glob("D:/ST8/data/single/rawData/patient*/*_FLAIR.nii.gz")
    groundTruth = glob.glob("D:/ST8/data/longitudinal/groundTruth/*.nii.gz")
    atlas = sitk.ReadImage('D:/ST8/data/longitudinal/atlas/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii.gz', sitk.sitkFloat32)
    fileListFLAIR_SS = glob.glob("D:/ST8/data/single/brainExtracted/skullStripped/*_FLAIR_SkullStripped.nii.gz")

     gADFilter(fileListFLAIR_SS[0], 'D:/ST8/data/single/ss_filtered/patient01_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[1], 'D:/ST8/data/single/ss_filtered/patient02_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[2], 'D:/ST8/data/single/ss_filtered/patient03_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[3], 'D:/ST8/data/single/ss_filtered/patient04_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[4], 'D:/ST8/data/single/ss_filtered/patient05_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[5], 'D:/ST8/data/single/ss_filtered/patient06_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[6], 'D:/ST8/data/single/ss_filtered/patient07_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[7], 'D:/ST8/data/single/ss_filtered/patient08_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[8], 'D:/ST8/data/single/ss_filtered/patient09_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[9], 'D:/ST8/data/single/ss_filtered/patient10_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[10], 'D:/ST8/data/single/ss_filtered/patient11_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[11], 'D:/ST8/data/single/ss_filtered/patient12_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[12], 'D:/ST8/data/single/ss_filtered/patient13_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[13], 'D:/ST8/data/single/ss_filtered/patient14_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[14], 'D:/ST8/data/single/ss_filtered/patient15_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[15], 'D:/ST8/data/single/ss_filtered/patient16_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[16], 'D:/ST8/data/single/ss_filtered/patient17_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[17], 'D:/ST8/data/single/ss_filtered/patient18_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[18], 'D:/ST8/data/single/ss_filtered/patient19_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[19], 'D:/ST8/data/single/ss_filtered/patient20_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[20], 'D:/ST8/data/single/ss_filtered/patient21_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[21], 'D:/ST8/data/single/ss_filtered/patient22_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[22], 'D:/ST8/data/single/ss_filtered/patient23_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[23], 'D:/ST8/data/single/ss_filtered/patient24_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[24], 'D:/ST8/data/single/ss_filtered/patient25_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[25], 'D:/ST8/data/single/ss_filtered/patient26_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[26], 'D:/ST8/data/single/ss_filtered/patient27_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[27], 'D:/ST8/data/single/ss_filtered/patient28_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[28], 'D:/ST8/data/single/ss_filtered/patient29_FLAIR_SkullStripped_GAD.nii.gz')
     gADFilter(fileListFLAIR_SS[29], 'D:/ST8/data/single/ss_filtered/patient30_FLAIR_SkullStripped_GAD.nii.gz')

    
    fileListFLAIRGAD = glob.glob("D:/ST8/data/single/ss_filtered/*_FLAIR_SkullStripped_GAD.nii.gz")
    
    n4_bias_field_correction(fileListFLAIRGAD[0], 'D:/ST8/data/single/n4/patient01_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[1], 'D:/ST8/data/single/n4/patient02_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[2], 'D:/ST8/data/single/n4/patient03_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[3], 'D:/ST8/data/single/n4/patient04_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[4], 'D:/ST8/data/single/n4/patient05_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[5], 'D:/ST8/data/single/n4/patient06_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[6], 'D:/ST8/data/single/n4/patient07_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[7], 'D:/ST8/data/single/n4/patient08_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[8], 'D:/ST8/data/single/n4/patient09_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[9], 'D:/ST8/data/single/n4/patient10_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[10], 'D:/ST8/data/single/n4/patient11_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[11], 'D:/ST8/data/single/n4/patient12_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[12], 'D:/ST8/data/single/n4/patient13_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[13], 'D:/ST8/data/single/n4/patient14_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[14], 'D:/ST8/data/single/n4/patient15_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[15], 'D:/ST8/data/single/n4/patient16_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[16], 'D:/ST8/data/single/n4/patient17_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[17], 'D:/ST8/data/single/n4/patient18_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[18], 'D:/ST8/data/single/n4/patient19_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[19], 'D:/ST8/data/single/n4/patient20_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[20], 'D:/ST8/data/single/n4/patient21_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[21], 'D:/ST8/data/single/n4/patient22_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[22], 'D:/ST8/data/single/n4/patient23_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[23], 'D:/ST8/data/single/n4/patient24_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[24], 'D:/ST8/data/single/n4/patient25_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[25], 'D:/ST8/data/single/n4/patient26_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[26], 'D:/ST8/data/single/n4/patient27_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[27], 'D:/ST8/data/single/n4/patient28_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[28], 'D:/ST8/data/single/n4/patient29_FLAIR_SkullStripped_GAD_N4.nii.gz')
    n4_bias_field_correction(fileListFLAIRGAD[29], 'D:/ST8/data/single/n4/patient30_FLAIR_SkullStripped_GAD_N4.nii.gz')
    
    return

main()

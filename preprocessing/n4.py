import os
import SimpleITK as sitk


def n4_bias_field_correction(input_file,
                             output_file,
                             masking_file=None,
                             number_iterations=[10, 40, 70, 120]):  # set the limit of iterations per resolution level,
    input_image = sitk.ReadImage(str(input_file))
    masking_image = sitk.OtsuThreshold(input_image, 0, 1,
                                    200)  # Masking mage input  specifies the pixels which are utilized to estimate the
    # bias-field and suppress pixels close to zero.
    input_image = sitk.Cast(input_image,
                          sitk.sitkFloat32)  # The sitk requires to have pixel type of either sitkFloat32 or sitkFloat64
    print("Input casted")
    correcting = sitk.N4BiasFieldCorrectionImageFilter() # adding th efilters which has to be used, adding function
    print("1")
    correcting.SetMaximumNumberOfIterations(number_iterations) # N4 is iterative adding the number of iteration prevously stated
    print("2")
    output_image = correcting.Execute(input_image, masking_image) # Correcting the image based on the previous parameters
    print("Done")
    #sitk.WriteImage(output_img, str(output_file))
    if masking_file:                    
        sitk.WriteImage(masking_image, str(masking_file)) # if we have a mask  svae the mask 

    sitk.WriteImage(output_image, output_file) # Save the bias field corrected image

    return os.path.abspath(output_file) # return image to to specified path 


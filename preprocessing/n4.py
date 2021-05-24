import os
import SimpleITK as sitk


def n4_bias_field_correction(input_file,
                             outputFile,
                             mask_file=None,
                             number_iterations=[10, 40, 70, 120]):  # set the limit of iterations per resolution level,
    input_img = sitk.ReadImage(str(input_file))
    mask_image = sitk.OtsuThreshold(input_img, 0, 1,
                                    200)  # MaskImage input which specifies which pixels are used to estimate the
    # bias-field and suppress pixels close to zero.
    input_img = sitk.Cast(input_img,
                          sitk.sitkFloat32)  # The  input is required to have pixel type of either sitkFloat32 or
    # sitkFloat64
    print("Input casted")
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    print("1")
    corrector.SetMaximumNumberOfIterations(number_iterations)
    print("2")
    output_img = corrector.Execute(input_img, mask_image)
    print("Done")
    #sitk.WriteImage(output_img, str(output_file))
    if mask_file:
        sitk.WriteImage(mask_image, str(mask_file))

    sitk.WriteImage(output_img, outputFile)

    return os.path.abspath(outputFile)


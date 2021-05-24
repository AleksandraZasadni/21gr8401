import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os

def gADFilter(inputFile, outputFile):
    class executingCommand(sitk.Command):
        def __init__(self, po):
            # required
            super(executingCommand, self).__init__()
            self.processObject = po

        def Execute(self):
            print(f"{self.processObject.GetName()} Progress: {self.processObject.GetProgress():1.2f}")


    # using simpleITK
    image = sitk.ReadImage(str(inputFile))
    # imgReader = sitk.ImageFileReader()
    # imgReader.SetFileName('D:/ST8/data/longitudinal/rawData/patient1/patient1_study1_FLAIR.nii.gz')
    # imgReader.SetImageIO("NiftiImageIO")
    # image = imgReader.Execute()

    # implements an $N$-dimensional version of the classic Perona-Malik anisotropic diffusion equation for scalar-valued images \cite{Perona1990}.
    # remove noise from digital images without blurring edges computation of the level set evolution.

    # Typical values for the time step are $0.25$ in $2D$ images and $0.125$ in $3D$ images.
    # The number of iterations is typically set to $5$; more iterations result in further smoothing and will increase
    # the computing time linearly

    pixelID = image.GetPixelID()
    numberOfIterations = 5  # default 10, time step used to discritize diffusion eq
    conductance = 0.02  # default 1, the lower, the stronger preservation of features
    timeStep = 0.04  # default 0.125 discritizes diffusion equation

    gADF = sitk.GradientAnisotropicDiffusionImageFilter()

    gADF.SetNumberOfIterations(numberOfIterations)
    gADF.SetTimeStep(timeStep)
    gADF.SetConductanceParameter(conductance)

    # Timing - can be removed if not needed
    gADF.AddCommand(sitk.sitkStartEvent, lambda: print("Start Event"))
    gADF.AddCommand(sitk.sitkEndEvent, lambda: print("End event"))

    cmd = executingCommand(gADF)
    gADF.AddCommand(sitk.sitkProgressEvent, cmd)

    image = gADF.Execute(sitk.Cast(image, sitk.sitkFloat32))
    image = sitk.Cast(image, sitk.sitkUInt8)

    # writing to an image
    imgWriter = sitk.ImageFileWriter()
    imgWriter.SetFileName('D:/ST8/data/longitudinal/rawData/patient1/patient1_study1_FLAIR_sitk.nii.gz')
    imgWriter.Execute(image)
    sitk.WriteImage(image, outputFile)

    return os.path.abspath(outputFile)

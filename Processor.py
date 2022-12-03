import custom_classes
import sys
import matplotlib.pyplot as plt


grayscale = custom_classes.grayscale
rgb = custom_classes.rgb

def check():
    '''
        check if any files are missing, and whether the user runs the program in the correct way
        return: input image, a 2D array
    '''

    # use pre-set figure, fontsize, style formatting
    try:
        plt.style.use('style.txt')
    except OSError:
        print("Style text file not found.", file=sys.stderr)
        sys.exit()

    # check if the number of command line arguments is correct, exit the program with error message if it is not
    if len(sys.argv) != 2:
        print(
            "Usage: interactive Fourier transform image processing of input image, works for both grayscale and RGB images.",
            file=sys.stderr)
        print("This program takes in image file name as argument.", file=sys.stderr)
        sys.exit()

    # check whether the image is a PNG file
    if sys.argv[1][-4:] != ".png":
        print("Image must be a PNG file", file=sys.stderr)
        sys.exit()

    # check if the image file exists, exit the program with error message if it does not exist
    try:
        img = plt.imread(sys.argv[1])
    except FileNotFoundError:
        print("Input image not found.", file=sys.stderr)
        sys.exit()

    return img



if __name__ == "__main__":

    # img = check()
    #
    # print("Initiating Program ...")
    #
    # #construct grayscale class if the image is a grayscale image,
    # #construct rgb class is the image is a rbg image
    # if img.ndim == 2:
    #     window = grayscale(img)
    # elif img.ndim == 3:
    #     window = rgb(img)

    print(os.__version__)
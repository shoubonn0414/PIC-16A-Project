# Interactive Fourier Transform Image Processor

Group Member: William Chang

## Project Description

This Python-built Fourier transform image processing interface allows the user to interact with the frequency spectrum of the input image.
- **Error Handling**
 
    The user runs the program from the terminal by passing an image file as an argument. The program first checks whether there is a correct number of argument been passed, whether the image is a PNG file, and whether the image file exists and handles the errors elegantly. The program then checks whether the image is a grayscale image or a RGB image, and construct class objects accordingly `grayscale` for grayscale images and class `rgb` for RGB images)

- **Fourier Transform**
 
    The underlying process of this program is inverse Fourier transforming the frequency spectrum many times. The frequency spectrum of the image is generated by Fourier transforming and centralizing the zero-components of the spectrum. For a grayscale image, the Fourier transform is done directly to its intensity; for a RGB image, the Fourier transform is each applied to its R, G, B channels. Every time after the user edited the frequency spectrum (via mouse editing or button editing), the spectrum is decentralized and inverse Fourier transformed to generate the processed image (for a RGB image, inverse Fourier transform is each applied to its R, G, B spectrum and stacked together to generate the processed image).

- **Mouse Editing**

    The user can directly edit the frequency spectrum, and in turn affect the processed image, by dragging areas of the spectrum to delete it. This is done by establishing connections between mouse events and the callback function. The program stores the mouse coordinates where the click and the release (defines an area) happens and delete the corresponding area of the spectrum.
		
- **Image Filters**

    The user can also choose to apply a default Gaussian low/high pass filter to edit the frequency spectrum via buttons on the window. Such function is achieved by linking the button click events to the callback function.
    - **Gaussian filters**
		
		The low pass filter and the high pass filter are defined as
        $$G_L(x,y) = e^{-\frac{D(x,y)^2}{2C^2}} \quad\quad G_H(x,y) = 1-e^{-\frac{D(x,y)^2}{2C^2}}$$
      where $D(x,y)$ is the distance of each point from the center of the image and $C$ is an arbitrary constant that corresponds to the standard deviation of the Gaussian function.
			
- **Other Features**

    The program also allows the user to undo the changes to the spectrum, save the processed image, and exit the program via the buttons on the window.
    - **Undo the Changes**
		
		The undoing is done by creating a copy of the frequency spectrum at the start of the program and replacing the edited spectrum with the copy.
    - **Save Processed Images**
		
		The program creates a directory named "figures" if it does not exist, captures the processed image, and saves the image in ascending file names to prevent duplicate of file names.
    - **Exit the Program** 
		
		The program closes the window and kills the program


### List of Python Packages
- `numpy` (1.22.3)
- `matplotlib` (3.5.1)
- `sys` (3.10)
- `os` (3.10.8)

###Detailed Description of the Demo File

- **Instructions to Run the Program**

    - The program takes in a second command line argument other than the Python file name itself. From the terminal, type `python3 Processor.py ImageName.png`, where `ImageName.png` is the file name/file path of the image that the user wants to import to the program
    - There are also two example images provided that can be used to test the program
        - `grayscale_example.png`: a grayscale image
        - `rgb_example.png`: a RGB image
    - NOTE: If the program is run without a second command line argument, a standard error will appear

        > Usage: interactive Fourier transform image processing of input image, works for both grayscale and RGB images.
				
        > This program takes in image file name as argument.

- **Processor Interface**

    After the program is initiated, a interface such as the following will be created:
    
    <p>
       <img src="https://github.com/shoubonn0414/PIC-16A-Project/blob/main/README%20materials/window.png" width="400" />
	<figcaption><b>Fig. 1: User Interface</b></figcaption>
    </p>

    The interface contains the following features:
        
	- **Display**
            
	    - Input Image: does not change
	    - Processed Image: changes in real-time when the frequency spectrum of the original image is changed
	    
	- Interactions
	    - Workspace of Frequency Spectrum: The user can drag parts of the spectrum with their mouse that they want to delete
	    - Buttons
	      - `Undo`: undo all changes to the frequency spectrum
	      - `Low Pass Filter`: apply Gaussian low pass filter to the spectrum
	      - `High Pass Filter`: apply Gaussian high pass filter to the spectrum
	      - `Save Image`: save the processed image in a directory named `saved_figure`
	      - `Exit`: close the window and exit the program
	      
    The user can repeatedly apply the filters or/and delete parts of the spectrum to see how the processed image is affected. Here are some examples of the filters applied to the image.
    
    <p>
       <img src="https://github.com/shoubonn0414/PIC-16A-Project/blob/main/README%20materials/high_pass.png" title="This is a Title" width="400"/>
	<figcaption><b>Fig. 2: Gaussian High Pass Filter Applied</b></figcaption>
    </p>

    <p>
       <img src="https://github.com/shoubonn0414/PIC-16A-Project/blob/main/README%20materials/filter.png" width="400" />
	<figcaption><b>Fig. 3: Gaussian High Pass Filter Applied after Gaussian Low Pass Filter is Applied</b></figcaption>
    </p>
		
### Scope and Limitations

This program allows the image to be processed in real time whenever the frequency spectrum is altered. As the fourier transform of the image is done pixel-by-pixel in the background, the large the image size, the slower the program runs. Therefore, a certain amount of computing power is required for this program.

For future extensions, we can consider adding different filters to the program, such as smoothing and sharpening filters. We can even allow the user to customize the parameters of these filters. This will further shed light into how the frequency spectrum actually affects the image. 

### References and Acknowledgement

Image Fourier Transform Algorithm selected from: D. Ballard and C. Brown Computer Vision, Prentice-Hall, 1982, pp 24 - 30.

### Software Demo Video

Grayscale Image:
https://user-images.githubusercontent.com/119761305/205468357-baecc8c7-bc2c-4abc-b960-17474352c08e.mp4

RGB Image:
https://user-images.githubusercontent.com/119761305/205468358-8ff4d774-70fc-47dd-98c0-39851d53e357.mp4

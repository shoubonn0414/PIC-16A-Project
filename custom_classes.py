#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.widgets import Button


class grayscale():
    '''
        class called when the processor input is a grayscale image, namely the image dimension is 2

        architecture:
            1. Initialize variables
                - apply fourier transform to the image
                - define constants
                - create a window containing subplots
                - create buttons in the window
            2. Define mouse events
                - mouse connection
                - detect mouse click
                - detect mouse release
                - update the Fourier spectrum and the processed image
            3. Define filters
                - Gaussian High Pass Filter
                - Gaussian Low Pass Filter
            4. Update, save figure, and exit the program:
                - image updated after every mouse event
                - save image if wanted
                - kills the program if wanted
    '''

    def __init__(self, image):
        '''
            initialize constants, create window and buttons, and fourier transform the image

            arg: image to be transformed, a 2D array
        '''


        self.img = image
        # flip image upside down to allow the origin be at lower left
        self.img = np.flipud(self.img)

        # apply fast fourier transform to image,
        # and shift the zero-frequency spectrum to the center to allow better comprehension
        self.spectrum = np.fft.fftshift(np.fft.fft2(self.img))

        # inverse-shifting the spectrum, and apply inverse fourier transform to the spectrum
        self.transformedImg = np.fft.ifft2(np.fft.ifftshift(self.spectrum))

        # create a copy for the frequency spectrum to allow later restoration of spectrum
        self.copy = np.fft.fftshift(np.fft.fft2(self.img))

        # initiate bounds, later used to alter the frequency spectrum array
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = 0, 0

        # constant used in Guassian filters, corresponds to the standard deviation of the Gaussian distribution
        self.C = 30

        # initiate saved figure name from 1
        self.figCount = 1


        # WINDOW
        #######################################################################################################

        # create a grid with 2 rows and 10 columns with specified width and height space to situate the figure
        grid = plt.GridSpec(2, 10, wspace=10, hspace=0.3)

        # create figure with specified figure size and title name
        self.fig = plt.figure(figsize=(15,9), num="GrayScale Image Processing")

        # put original image (ax1) at upper left, processed image (ax3) at upper right,
        # and frequency spectrum workspace (ax2) at bottom
        self.ax1 = plt.subplot(grid[0, :5])
        self.ax3 = plt.subplot(grid[0, 5:])
        self.ax2 = plt.subplot(grid[1, :8])
        #######################################################################################################


        #BUTTONS
        # create buttons on the figure and link them to the functions when clicked
        # customize buttons with axes[left, buttom, width, height]
        #######################################################################################################

        # UNDO button
        self.UNDO = Button(plt.axes([0.63, 0.37, 0.07, 0.063]), "Undo", hovercolor="greenyellow")
        self.UNDO.on_clicked(self.undo)

        # Low Pass Filter button
        self.GLP = Button(plt.axes([0.63, 0.30, 0.13, 0.063]), "Low Pass Filter", hovercolor="greenyellow")
        self.GLP.on_clicked(self.glp)

        # High Pass Filter button
        self.GHP = Button(plt.axes([0.63, 0.23, 0.13, 0.063]), "High Pass Filter", hovercolor="greenyellow")
        self.GHP.on_clicked(self.ghp)

        # SAVE button
        self.SAVE = Button(plt.axes([0.63, 0.16, 0.10, 0.063]), "Save Image", hovercolor="greenyellow")
        self.SAVE.on_clicked(self.saveFig)

        # EXIT button
        self.EXIT = Button(plt.axes([0.63, 0.09, 0.07, 0.063]), "Exit", color="crimson")
        self.EXIT.on_clicked(self.exit)
        #######################################################################################################

        # draw the initial image, spectrum, and processed image
        self.draw(self.img, self.spectrum, self.transformedImg)

        # establish connection between canvas and mouse events
        self.connect()

        plt.show()


    def undo(self, event):
        '''
            undo any changes done to the frequency spectrum
            arg: event name
        '''

        # overwrite the frequency spectrum with the copy
        self.spectrum = self.copy

        # create the copy again
        self.copy = np.fft.fftshift(np.fft.fft2(self.img))

        print("Image Restored.")

        # draw the figure again
        self.draw(self.img, self.spectrum, self.img)


    # MOUSE EVENTS
    #######################################################################################################

    def connect(self):
        '''
            establishing connection between mouse events and callback functions
        '''
        self.cidclick = self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.release)

    def click(self, event):
        '''
            called when mouse is clicked, and the check the properties of the mouse click
            arg: event name
        '''

        # check if the mouse click happens in the frequency spectrum figure (ax2)
        if event.inaxes != self.ax2:
            return
        # check if the click is a left mouse click
        if event.button == 1:
            # set first bounds to where the mouse click is
            self.x1, self.y1 = int(event.xdata), int(event.ydata)

    def release(self, event):
        '''
            called when mouse is released, and the check the properties of the mouse release
            arg: event name
        '''

        # check if the mouse click happens in the frequency spectrum figure (ax2)
        if event.inaxes != self.ax2:
            return
        # check if the click is a left mouse click
        if event.button == 1:
            # set second bounds to where the mouse click is
            self.x2, self.y2 = int(event.xdata), int(event.ydata)
        # call the update function when the area drag is complete
        self.update()

    def update(self):
        '''
            update the frequency spectrum, inverse transform the spectrum, and draw the spectrum
        '''

        # set the part of the frequency spectrum, where the mouse dragged area is, to 1,
        # essentially deleting the part of the spectrum
        self.spectrum[min(self.y1, self.y2):max(self.y1, self.y2), min(self.x1, self.x2):max(self.x1, self.x2)] = 1

        # inverse transform the updated spectrum
        self.transformedImg = np.fft.ifft2(np.fft.ifftshift(self.spectrum))

        # draw the figure again
        self.draw(self.img, self.spectrum, self.transformedImg)
    #######################################################################################################


    # FILTERS
    #######################################################################################################

    def dist(self, p1, p2):
        '''
            compute the 2D distance between p1 and p2
            arg:
                p1, p2: a tuple specifying the 2D coordinates of the points
        '''

        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def glp(self, event):
        '''
            middle-man function used to link the button press to the filter function
            arg: event name
        '''
        print("Applying Gaussian Low Pass Filter ...")
        self.gaussianLowPassFilter(self.C, self.img.shape)

    def gaussianLowPassFilter(self, C, imgShape):
        '''
            Gaussian Low Pass Filter: eliminates high frequencies
            arg: a tuple specifying the shape of the image
        '''

        filter = np.zeros(imgShape)

        # row, column, and the center of the image
        row, column = imgShape
        center = (row/2, column/2)

        # use meshgrid instead of nested for loop to speed up the processing
        x = np.arange(column)
        y = np.arange(row)
        xg, yg = np.meshgrid(x, y)

        # create the filter with the Guassian function
        filter[yg, xg] = np.exp(((-self.dist((yg, xg), center) ** 2) / (2 * (C ** 2))))

        # apply filter to the frequency spectrum
        self.spectrum *= filter
        # inverse transform the updated spectrum
        self.transformedImg = np.fft.ifft2(np.fft.ifftshift(self.spectrum))

        # draw the figure again
        self.draw(self.img, self.spectrum, self.transformedImg)

    def ghp(self, event):
        '''
            middle-man function used to link the button press to the filter function
            arg: event name
        '''
        print("Applying Gaussian High Pass Filter ...")
        self.gaussianHighPassFilter(self.C, self.img.shape)

    def gaussianHighPassFilter(self, C, imgShape):
        '''
            Gaussian High Pass Filter: eliminates low frequencies
            arg: a tuple specifying the shape of the image
        '''
        filter = np.zeros(imgShape)

        # row, column, and the center of the image
        row, column = imgShape
        center = (row / 2, column / 2)

        # use meshgrid instead of nested for loop to speed up the processing
        x = np.arange(column)
        y = np.arange(row)
        xg, yg = np.meshgrid(x, y)

        # create the filter with the Guassian function
        filter[yg, xg] = 1 - np.exp(((-self.dist((yg, xg), center) ** 2) / (2 * (C ** 2))))

        # apply filter to the frequency spectrum
        self.spectrum *= filter
        # inverse transform the updated spectrum
        self.transformedImg = np.fft.ifft2(np.fft.ifftshift(self.spectrum))

        #draw the figure again
        self.draw(self.img, self.spectrum, self.transformedImg)
    #######################################################################################################

    def draw(self, image, transformed, newImage):
        '''
            draw the original image, frequency spectrum, and transfromed image
            args:
                image: the original image, a 2D array
                transformed: the frequency spectrum of the original image, a 2D array
                newImage: the transformed image, a 2D array
        '''

        # draw original image with origin at lower left
        self.ax1.imshow(image, "gray", origin="lower"), self.ax1.set_title("Original Image")

        # draw frequency spectrum with origin at lower left
        # (with absolute value to obtain the intensity and log scale to better visualize)
        self.ax2.imshow(np.log(1+np.abs(transformed)), "gray", origin="lower"), self.ax2.set_title("Centralized Frequency Spectrum")

        # draw transformed image image with origin at lower left
        # (with absolute value to obtain the intensity)
        self.ax3.imshow(np.abs(newImage), "gray", origin="lower"), self.ax3.set_title("Processed Image")

        # update the canvas
        self.fig.canvas.flush_events()
        self.fig.canvas.draw_idle()

        print("Image Updated.")
        print("##################################")

    def saveFig(self, event):
        '''
            save processed image when called
            arg: event name
        '''

        # save just the window portion the ax3's boundaries
        extent = self.ax3.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())

        # create path to new directory
        folder_path = str(os.getcwd())+'/saved_figures'

        # create directory at the path if the folder does not yet exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # if there are any images already saved in the directory, make sure that the image saved has the highest count
        while os.path.exists(folder_path+'figure'+str(self.figCount)+'.png'):
            self.figCount += 1

        # increase the saved area by 10% in the x-direction and 20% in the y-direction
        self.fig.savefig(str(os.getcwd())+'/saved_figures/figure'+str(self.figCount)+'.png', bbox_inches=extent.expanded(1.25, 1.25))

        # increate the figure count by 1 to prevent two figures having same name
        self.figCount += 1

        print("Image Saved.")

    def exit(self, event):
        '''
            exit the program when called
            arg: event name
        '''
        print("Program exited. Thanks for using.")
        sys.exit()


class rgb():

    '''
        class called when the processor input is a RGB image, namely the image dimension is 3

        architecture:
            1. Initialize variables
                - apply fourier transform to the image
                - define constants
                - create a window containing subplots
                - create buttons in the window
            2. Define mouse events
                - mouse connection
                - detect mouse click
                - detect mouse release
                - update the Fourier spectrum and the processed image
            3. Define filters
                - Gaussian High Pass Filter
                - Gaussian Low Pass Filter
            4. Update, save figure, and exit the program:
                - image updated after every mouse event
                - save image if wanted
                - kills the program if wanted
    '''

    def __init__(self, image):
        '''
            initialize constants, create window and buttons, and fourier transform the image

            arg: image to be transformed, a 2D array
        '''

        self.img = image
        # flip image upside down to allow the origin be at lower left
        self.img = np.flipud(self.img)

        # apply fast fourier transform to R, G, B channels of the image
        # and shift the zero-frequency spectrum to the center to allow better comprehension
        self.spectrum = np.array([np.fft.fftshift(np.fft.fft2((self.img[:, :, 0]))),
                                  np.fft.fftshift(np.fft.fft2((self.img[:, :, 1]))),
                                  np.fft.fftshift(np.fft.fft2((self.img[:, :, 2])))])

        # inverse-shifting the R, G, B channels, stack them together
        # and apply inverse fourier transform to the spectrum
        self.transformedImg = np.dstack([np.fft.ifft2(np.fft.ifftshift(self.spectrum[0])),
                                         np.fft.ifft2(np.fft.ifftshift(self.spectrum[1])),
                                         np.fft.ifft2(np.fft.ifftshift(self.spectrum[2]))])

        # create a copy for the frequency spectrum to allow later restoration of spectrum
        self.copy = np.array([np.fft.fftshift(np.fft.fft2((self.img[:, :, 0]))),
                              np.fft.fftshift(np.fft.fft2((self.img[:, :, 1]))),
                              np.fft.fftshift(np.fft.fft2((self.img[:, :, 2])))])

        # initiate bounds, later used to alter the frequency spectrum array
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = 0, 0

        #constant used in Guassian filters, corresponds to the standard deviation of the Gaussian distribution
        self.C = 30

        #initiate saved figure name from 1
        self.figCount = 1

        # WINDOW
        #######################################################################################################

        # create a grid with 2 rows and 10 columns with specified width and height space to situate the figure
        grid = plt.GridSpec(2, 10, wspace=10, hspace=0.3)

        # create figure with specified figure size and title name
        self.fig = plt.figure(figsize=(15,9), num="RGB Image Processing")

        # put original image (ax1) at upper left, processed image (ax3) at upper right,
        # and frequency spectrum workspace (ax2) at bottom
        self.ax1 = plt.subplot(grid[0, :5])
        self.ax3 = plt.subplot(grid[0, 5:])
        self.ax2 = plt.subplot(grid[1, :8])
        #######################################################################################################


        #BUTTONS
        # create buttons on the figure and link them to the functions when clicked
        # customize buttons with axes[left, buttom, width, height]
        #######################################################################################################

        # UNDO button
        self.UNDO = Button(plt.axes([0.63, 0.37, 0.07, 0.063]), "Undo", hovercolor="greenyellow")
        self.UNDO.on_clicked(self.undo)

        # Low Pass Filter button
        self.GLP = Button(plt.axes([0.63, 0.30, 0.13, 0.063]), "Low Pass Filter", hovercolor="greenyellow")
        self.GLP.on_clicked(self.glp)

        # High Pass Filter button
        self.GHP = Button(plt.axes([0.63, 0.23, 0.13, 0.063]), "High Pass Filter", hovercolor="greenyellow")
        self.GHP.on_clicked(self.ghp)

        # SAVE button
        self.SAVE = Button(plt.axes([0.63, 0.16, 0.10, 0.063]), "Save Image", hovercolor="greenyellow")
        self.SAVE.on_clicked(self.saveFig)

        # EXIT button
        self.EXIT = Button(plt.axes([0.63, 0.09, 0.07, 0.063]), "Exit", color="red")
        self.EXIT.on_clicked(self.exit)
        #######################################################################################################

        # draw the initial image, spectrum, and processed image
        self.draw(self.img, self.spectrum[0], self.transformedImg)

        # establish connection between canvas and mouse events
        self.connect()

        plt.show()

    def undo(self, event):
        '''
            undo any changes done to the frequency spectrum
            arg: event name
        '''

        # overwrite the frequency spectrum with the copy
        self.spectrum = self.copy

        # create the copy again
        self.copy = np.array([np.fft.fftshift(np.fft.fft2((self.img[:, :, 0]))),
                              np.fft.fftshift(np.fft.fft2((self.img[:, :, 1]))),
                              np.fft.fftshift(np.fft.fft2((self.img[:, :, 2])))])

        print("Image Restored.")

        # draw the figure again
        self.draw(self.img, self.spectrum[0], self.img)


    # MOUSE EVENTS
    #######################################################################################################

    def connect(self):
        '''
            establishing connection between mouse events and callback functions
        '''
        self.cidclick = self.fig.canvas.mpl_connect('button_press_event', self.click)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.release)

    def click(self, event):
        '''
            called when mouse is clicked, and the check the properties of the mouse click
            arg: event name
        '''

        # check if the mouse click happens in the frequency spectrum figure (ax2)
        if event.inaxes != self.ax2:
            return
        # check if the click is a left mouse click
        if event.button == 1:
            # set first bounds to where the mouse click is
            self.x1, self.y1 = int(event.xdata), int(event.ydata)

    def release(self, event):
        '''
            called when mouse is released, and the check the properties of the mouse release
            arg: event name
        '''

        # check if the mouse click happens in the frequency spectrum figure (ax2)
        if event.inaxes != self.ax2:
            return
        # check if the click is a left mouse click
        if event.button == 1:
            # set second bounds to where the mouse click is
            self.x2, self.y2 = int(event.xdata), int(event.ydata)
        # call the update function when the area drag is complete
        self.update()

    def update(self):
        '''
            update the frequency spectrum, inverse transform the spectrum, and draw the spectrum
        '''

        # set R, G, B channels of the part of the frequency spectrum, where the mouse dragged area is, to 0,
        # essentially deleting the part of the spectrum
        self.spectrum[0][min(self.y1, self.y2):max(self.y1, self.y2), min(self.x1, self.x2):max(self.x1, self.x2)] = 0
        self.spectrum[1][min(self.y1, self.y2):max(self.y1, self.y2), min(self.x1, self.x2):max(self.x1, self.x2)] = 0
        self.spectrum[2][min(self.y1, self.y2):max(self.y1, self.y2), min(self.x1, self.x2):max(self.x1, self.x2)] = 0

        # inverse transform the updated spectrum
        self.transformedImg = np.dstack([np.fft.ifft2(np.fft.ifftshift(self.spectrum[0])),
                                         np.fft.ifft2(np.fft.ifftshift(self.spectrum[1])),
                                         np.fft.ifft2(np.fft.ifftshift(self.spectrum[2]))])

        # draw the figure again
        self.draw(self.img, self.spectrum[0], self.transformedImg)
    #######################################################################################################


    # FILTERS
    #######################################################################################################

    def dist(self, p1, p2):
        '''
            compute the 2D distance between p1 and p2
            arg:
                p1, p2: a tuple specifying the 2D coordinates of the points
        '''

        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def glp(self, event):
        '''
            middle-man function used to link the button press to the filter function
            arg: event name
        '''
        print("Applying Gaussian Low Pass Filter ...")
        self.gaussianLowPassFilter(self.C, self.img.shape[:2])

    def gaussianLowPassFilter(self, C, imgShape):
        '''
            Gaussian Low Pass Filter: eliminates high frequencies
            arg: a tuple specifying the shape of the image
        '''
        filter = np.zeros(imgShape)

        # row, column, and the center of the image
        row, column = imgShape
        center = (row/2, column/2)

        # use meshgrid instead of nested for loop to speed up the processing
        x = np.arange(column)
        y = np.arange(row)
        xg, yg = np.meshgrid(x, y)

        # create the filter with the Guassian function
        filter[yg, xg] = np.exp(((-self.dist((yg, xg), center) ** 2) / (2 * (C ** 2))))

        # apply filter to the frequency spectrum
        self.spectrum *= filter
        # inverse transform the updated spectrum
        self.transformedImg = np.dstack([np.fft.ifft2(np.fft.ifftshift(self.spectrum[0])),
                                         np.fft.ifft2(np.fft.ifftshift(self.spectrum[1])),
                                         np.fft.ifft2(np.fft.ifftshift(self.spectrum[2]))])

        # draw the figure again
        self.draw(self.img, self.spectrum[0], self.transformedImg)

    def ghp(self, event):
        '''
            middle-man function used to link the button press to the filter function
            arg: event name
        '''
        print("Applying Gaussian High Pass Filter ...")
        self.gaussianHighPassFilter(self.C, self.img.shape[:2])

    def gaussianHighPassFilter(self, C, imgShape):
        '''
            Gaussian High Pass Filter: eliminates low frequencies
            arg: a tuple specifying the shape of the image
        '''
        filter = np.zeros(imgShape)

        # row, column, and the center of the image
        row, column = imgShape
        center = (row / 2, column / 2)

        # use meshgrid instead of nested for loop to speed up the processing
        x = np.arange(column)
        y = np.arange(row)
        xg, yg = np.meshgrid(x, y)

        # create the filter with the Guassian function
        filter[yg, xg] = 1 - np.exp(((-self.dist((yg, xg), center) ** 2) / (2 * (C ** 2))))

        # apply filter to the frequency spectrum
        self.spectrum *= filter
        # inverse transform the updated spectrum
        self.transformedImg = np.dstack([np.fft.ifft2(np.fft.ifftshift(self.spectrum[0])),
                                         np.fft.ifft2(np.fft.ifftshift(self.spectrum[1])),
                                         np.fft.ifft2(np.fft.ifftshift(self.spectrum[2]))])

        #draw the figure again
        self.draw(self.img, self.spectrum[0], self.transformedImg)
    #######################################################################################################

    def draw(self, image, transformed, newImage):
        '''
            draw the original image, frequency spectrum, and transfromed image
            args:
                image: the original image, a 2D array
                transformed: the frequency spectrum of the original image, a 2D array
                newImage: the transformed image, a 2D array
        '''

        # draw original image with origin at lower left
        self.ax1.imshow(image, origin="lower"), self.ax1.set_title("Original Image")

        # draw frequency spectrum with origin at lower left
        # (with absolute value to obtain the intensity and log scale to better visualize)
        self.ax2.imshow(np.log(1 + np.abs(transformed)), "gray", origin="lower"), self.ax2.set_title(
            "Centralized Frequency Spectrum")

        # draw transformed image image with origin at lower left
        # (with absolute value to obtain the intensity)
        self.ax3.imshow(np.abs(newImage), origin="lower"), self.ax3.set_title("Processed Image")

        # update the canvas
        self.fig.canvas.flush_events()
        self.fig.canvas.draw_idle()

        print("Image Updated.")
        print("##################################")

    def saveFig(self, event):
        '''
            save processed image when called
            arg: event name
        '''
        # save just the window portion the ax3's boundaries
        extent = self.ax3.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())

        # create path to new directory
        folder_path = str(os.getcwd()) + '/saved_figures'

        # create directory at the path if the folder does not yet exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # if there are any images already saved in the directory, make sure that the image saved has the highest count
        while os.path.exists(folder_path + '/figure' + str(self.figCount) + '.png'):
            self.figCount += 1

        # increase the saved area by 10% in the x-direction and 20% in the y-direction
        self.fig.savefig(str(os.getcwd()) + '/saved_figures/figure' + str(self.figCount) + '.png',
                         bbox_inches=extent.expanded(1.25, 1.25))

        # increate the figure count by 1 to prevent two figures having same name
        self.figCount += 1

        print("Image Saved.")

    def exit(self, event):
        '''
            exit the program when called
            arg: event name
        '''
        print("Program exited. Thanks for using.")
        sys.exit()

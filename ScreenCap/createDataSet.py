import os
from PIL import Image
import numpy
#directory = "train20X20"
directory = "train25X25"
directory = "predict"
image_extension = ".png"
datasetFilename = directory+".csv"
import pandas
import csv

def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, 'r')
    image = image.convert("L")
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == 'RGB':
        channels = 3
    elif image.mode == 'L':
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = numpy.array(pixel_values).reshape((width, height, channels))
    return pixel_values

def addToCSVFile(filewriter,list):
    filewriter.writerow(list)

def downSizeImage(path):
    foo = Image.open(path)
    foo = foo.resize((96,54),Image.ANTIALIAS)
    foo.save(path, quality=95)

def changeTrainImagesPixel():
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(image_extension):
                npArray = numpy.asarray(get_image(filepath))
                #print "========================================================================="
                #print filepath
                downSizeImage(filepath)
                #print "========================================================================="

def createDataSet():
    csvfile = open(datasetFilename, 'wb')
    flag = False
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(image_extension):
                npArray = numpy.asarray(get_image(filepath))
                #print "========================================================================="
                a = npArray.flatten()
                label = file.split(".")[0]
                #print label.split("_")[0]
                image_class = label.split("_")[0]
                title_row =[]
                if flag == False :
                    pixelLen = len(a)
                    for i in range(pixelLen):
                        title_row.append("pixel"+str(i))
                    addToCSVFile(filewriter, ["label"] + title_row)
                    flag = True
                addToCSVFile(filewriter,[image_class]+list(a))
                #print "========================================================================="
    csvfile.close()

#changeTrainImagesPixel()
createDataSet()


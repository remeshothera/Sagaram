###############################################################
#    Author : Remeshkumar K K
#    Date : 04/19/2018
#    Main code which will start video and call other modules for ml , dataset creation.
###############################################################

import os
import pyautogui
import time
import threading
import os
import createDataSet
import svm
model = None
t_end = time.time() + 10 * 1

def play_video():
    print "Playing video.................."
    os.system('"vlc.exe --no-qt-privacy-ask --fullscreen --play-and-exit textinmotion.mp4"')

def startAnalysis():

    t1 = threading.Thread(target=play_video, args=[])
    t1.start()
    # Save the image
    i = 0
    time.sleep(5)
    while time.time() < t_end:
        pic = pyautogui.screenshot()
        pic.save("predict/" + "Screenshot_"+str(i)+".png" )
        i += 1


def initializeMachine():
    model = svm.executeMachine()
    return model

if __name__ == "__main__":
    #Initialize the model - SVM classifier
    model = initializeMachine()
    #Play Video and start analysis
    startAnalysis()
    createDataSet.changeTrainImagesPixel() # downscale the captured 1080p image
    createDataSet.createDataSet() # create dataset for prediction
    svm.classifyImage(model) # classify the image type : Good OR Corrupted
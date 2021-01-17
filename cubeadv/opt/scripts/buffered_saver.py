import numpy as np
import os

import cv2  # for resizing image

import sys

class BufferedImageSaver:
    """
    Stores incoming data in a Numpy ndarray and saves the array to disk once
    completely filled.
    """
    # rows = WINDOW_HEIGHT = 180
    # cols = WINDOW_HEIGHT = 320
    def __init__(self, filename: str, size: int,
                 rows: int, cols: int, depth:int, sensorname: str):
        """An array of shape (size, rows, cols, depth) is created to hold
        incoming images (this is the buffer). `filename` is where the buffer
        will be stored once full.
        """
        self.filename = filename + sensorname + '/'
        self.size = size
        self.sensorname = sensorname
        #dtype = np.float32 if self.sensorname == 'CameraDepth' else np.uint8
        
        ## dtype
        dtype = np.uint8 if (self.sensorname == 'CameraRGB_0' or self.sensorname == 'CameraRGB_1' 
            or self.sensorname == 'CameraRGB_2' or self.sensorname == 'CameraRGB_3') else np.float32
        # dtype = np.float32 if (self.sensorname == 'Lidar' or self.sensorname == 'Control') 
        #     else np.uint8
        
        self.buffer = np.zeros(shape=(size, rows, cols, depth),
                               dtype=dtype)
        self.index = 0
        self.reset_count = 0  # how many times this object has been reset
        
        ## check for syncrnization failures
        self.saved_count = 0

    def is_full(self):
        """A BufferedImageSaver is full when `self.index` is one less than
        `self.size`.
        """
        return self.index == self.size

    def reset(self):
        self.buffer = np.zeros_like(self.buffer)
        self.index = 0
        self.reset_count += 1

    def save(self):
        save_name = self.filename + str(self.reset_count) + '.npy'
        
        # make the enclosing directories if not already present
        folder = os.path.dirname(save_name)
        if not os.path.isdir(folder):
            os.makedirs(folder)
            
        self.saved_count += 1
        if(self.saved_count == self.reset_count + 1):
            # save the buffer
            np.save(save_name, self.buffer[:self.index + 1])
            print(self.size, " images saved in " + save_name)
            self.reset()
            return True
        else:
            print("frame skiped -----------------!!!!!!")
            #sys.exit(100)
            return False
            
    @staticmethod
    def process_by_type(img_bytes, name, buffer_rows, buffer_cols):
        """Converts the raw image to a more efficient processed version
        useful for training. The processing to be applied depends on the
        sensor name, passed as the second argument.
        """
        if name == 'CameraRGB_0' or name == 'CameraRGB_1' or name == 'CameraRGB_2' or name == 'CameraRGB_3':
            return img_bytes
        elif name == 'Lidar':
            return img_bytes.reshape(-1, 1, 3)  # 8000 x 1 x 3
        elif name == 'Coloured_Lidar':
            return img_bytes.reshape(-1, 1, 6)  # 8000 x 1 x 6
        elif name == "Control":
            return img_bytes.reshape(1, 1, 4)   # 1 x 1 x 4
        elif name == 'Spherical':
            return img_bytes.reshape(32, -1, 1) # 100 x 32 x 1
        elif name == 'Coloured_Spherical':
            return img_bytes.reshape(32, -1, 4) # 100 x 32 x 4
        elif name == 'All':
            return img_bytes.reshape(32, -1, 9) # 100 x 32 x 9
        elif name == 'Rotation':
            return img_bytes.reshape(1, 1, 3)
        elif name == 'Position':
            return img_bytes.reshape(1, 1, 3)
        elif name == 'Steer':
            return img_bytes.reshape(1, 1, 1)
        elif name == "Waypoint":
            return img_bytes.reshape(1, 1, 3)
        else:
            print("add_image saver name illegal")
            return None

    def add_image(self, img_bytes, name):
        """Save the current buffer to disk and reset the current object
        if the buffer is full, otherwise store the bytes of an image in
        self.buffer.
        """
        if self.is_full():
            success = self.save()
            if success:
                return self.add_image(img_bytes, name)
            else:
                return False
        else:
            #print('save')
            
            rows = self.buffer.shape[1]
            cols = self.buffer.shape[2]
            
            raw_image = self.process_by_type(img_bytes, name, rows, cols)            
            # print(raw_image.shape)      #      (1888, 1, 3)
            # print(self.buffer.shape)    #(1000, 3000, 1, 3)

            # for Lidar: (0, 0, 0) will be the place holder in the buffer indicating an empty detection
            self.buffer[self.index][:raw_image.shape[0]] = raw_image
            #print(self.buffer[self.index])
            self.index += 1
            return True
            
            
            
            
            
            
            
            

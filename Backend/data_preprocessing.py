import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# My utility functions/constants
from utility import *


# create a dictionary mapping the actions to numerical values
label_map = {label:num for num, label in enumerate(actions)}


# storing each frame in a folder eg. 'hello/1' into a curr_video var which in turn is stored in a single list containing all videos
list_of_all_videos, labels = [], []
for action in actions:
    for video_number in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        curr_video = []
        for frame_num in range(1, no_of_frames_in_a_video + 1):
            curr_frame = np.load(os.path.join(DATA_PATH, action, str(video_number), "{}.npy".format(frame_num)))
            curr_video.append(curr_frame)
        list_of_all_videos.append(curr_video)
        labels.append(label_map[action])


# demo - checking the number of videos, frames in each video and total keypoints in each frame
# np.array(sequences).shape
# (240, 30, 1662)   # If there are 8 actions with 30 videos each then, the shape represents that there are 240 videos with 30 frames and 1662 keypoints

# demo - checking the number of labels for total data
# np.array(labels).shape    # If there are 8 actions with 30 videos each then, the shape represents that there are 240 videos each with one label hence there are 240 labels
# (240,)


# store data as an numpy array for pre-processing
X = np.array(list_of_all_videos)

# store labels categorically eg. {'A': 0, 'B': 1, 'C': 2} is a dictionary then the value of 'y' will be (90, 3) for 90 videos distributed in 3 folders
y = to_categorical(labels).astype(int)

# the value of y will look as:
# array([[1, 0, 0, 0, 0, 0, 0, 0],
#        [1, 0, 0, 0, 0, 0, 0, 0],
#        [1, 0, 0, 0, 0, 0, 0, 0],
#        ...,
#        [0, 0, 0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 0, 0, 1],
#        [0, 0, 0, 0, 0, 0, 0, 1]])


# train-test-split takes inputs as: Data, labels, test_size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


# X_train.shape
# (228, 30, 1662)

# X_test.shape
# (12, 30, 1662)

# y_train.shape
# (228, 8)

# y_test.shape
# (12, 8)
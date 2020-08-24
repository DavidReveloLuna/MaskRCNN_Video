import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage.util import img_as_float
from skimage import img_as_ubyte
import colorsys
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    
camera = cv2.VideoCapture(0)

while camera:
    ret, frame = camera.read()
    frame = cv2.resize(frame, (640, 480), interpolation = cv2.INTER_AREA)
    
    results = model.detect([frame], verbose=0)
    
    # Visualize results
    r = results[0]
    #visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
    #                         class_names, r['scores'],title='frame')
  
    
    N =  r['rois'].shape[0]
    boxes=r['rois']
    masks=r['masks']
    class_ids=r['class_ids']
    scores=r['scores']
    

       
    hsv = [(i / N, 1, 0.7) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    
    random.shuffle(colors)
    #print("N_obj:",N)
    masked_image = frame.astype(np.uint32).copy()
    
    for i in range(N):
        
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        #print(colors[i])

        color = list(np.random.random(size=3) * 256)

    
        mask = masks[:, :, i]
        alpha=0.5

        
        for c in range(3):
            masked_image[:, :, c] = np.where(mask == 1,
                                  masked_image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  masked_image[:, :, c])
            
        
        frame_obj=masked_image.astype(np.uint8)
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(frame_obj, (x1, y1), (x2, y2),color, 2)  
        
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label
        cv2.putText(frame_obj,caption,(int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        masked_image = frame_obj.astype(np.uint32).copy()
    
        

            
    cv2.imshow('frame', frame_obj)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break;
        
camera.release()
cv2.destroyAllWindows()
    
# # Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]

# for filename in file_names:
    
#     image = skimage.io.imread(os.path.join(IMAGE_DIR, filename))

#     # Run detection
#     results = model.detect([image], verbose=1)
    
#     # Visualize results
#     r = results[0]
#     visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                             class_names, r['scores'],title=filename)

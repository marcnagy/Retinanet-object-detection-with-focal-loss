import torch

BATCH_SIZE = 1 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Base image resolution transforms.
WIDTH=2688
HEIGHT=1520
NUM_EPOCHS = 75 # Number of epochs to train for.
NUM_WORKERS = 4 # Number of parallel workers for data loading.
LR = 0.005 # Initial learning rate. 
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Keep `resolutions=None` for not using multi-resolution training,
# else it will be 50% lower than base `RESIZE_TO`, then base `RESIZE_TO`, 
# and 50% higher than base `RESIZE_TO`
# RESOLUTIONS = [
#     (int(RESIZE_TO/2), int(RESIZE_TO/2)),
#     (int(RESIZE_TO/1.777), int(RESIZE_TO/1.777)),
#     (int(RESIZE_TO/1.5), int(RESIZE_TO/1.5)),
#     (int(RESIZE_TO/1.333), int(RESIZE_TO/1.333)),
#     (RESIZE_TO, RESIZE_TO),
#     (int(RESIZE_TO*1.333), int(RESIZE_TO*1.333)),
#     (int(RESIZE_TO*1.5), int(RESIZE_TO*1.5)),
#     (int(RESIZE_TO*1.777), int(RESIZE_TO*1.777)),
#     (int(RESIZE_TO*2), int(RESIZE_TO*2))
# ]
RESOLUTIONS = None

# Training images and XML files directory.
TRAIN_IMG = 'data/training/JPEGImages'
TRAIN_ANNOT = 'data/training/Annotations'
# Validation images and XML files directory.
VALID_IMG = 'data/validation/JPEGImages'
VALID_ANNOT = 'data/validation/Annotations'
CLASSES = [
    'Player',
    'Ball'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after creating the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Automatic Mixed Preicision?
AMP = True

# If kept None, it will be incremental as exp1, exp2,
# else it will be named provided.
PROJECT_NAME = 'retinanet object detection with focal loss'
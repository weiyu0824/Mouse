# File directories
DATASET_DIR = '/m-ent1/ent1/wylin6/mouse/dataset/'
DATA2D_DIR = '/m-ent1/ent1/wylin6/mouse/preprocess/'
DATA3D_DIR = '/m-ent1/ent1/wylin6/mouse/preprocess_3d/'
GUIDE_DIR = '/m-ent1/ent1/wylin6/mouse/guide/'
QUERY_FILE_PATH = '/m-ent1/ent1/wylin6/mouse/query.csv'
ONTOLOGY_DIR = '/m-ent1/ent1/wylin6/mouse/ontology/'


# Dataset meta information
TIMESTAMPS = ['E11.5', 'E13.5', 'E15.5', 'E18.5', 'P4', 'P14', 'P56']
SHAPES = [(40, 75, 70), (69, 109, 89), (65, 132, 94), (40, 43, 67), (50, 43, 77), (50, 40, 68), (58, 41, 67)]
IMG_SHAPE = (80, 144, 96) #(60, 96, 80)

# Other
DEVICE = 'cuda:3'
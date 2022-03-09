!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

# build carpart model
def build_carparts_model():
    # configuration for DentScratch
    carparts_cfg = get_cfg()
    carparts_cfg.merge_from_file(
        model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    carparts_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    carparts_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
    carparts_cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    carparts_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
#     carparts_cfg.MODEL.WEIGHTS = '../input/carpartsdetection/CarPartsDetection.pth'
    # build the predictor
    carparts_predictor = DefaultPredictor(carparts_cfg)
    return carparts_predictor
carparts_predictor = build_carparts_model()


state_dict = torch.load("/content/CarPartsDetection.pth", map_location="cuda")
carparts_predictor.model.load_state_dict(state_dict['CarPartsDetection'])

def resize_image(image, scale_percent = 30):
    """
    Reduce image to 30%
    """
    # make a copy
    img = image.copy()
    #scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized_image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized_image


im = cv2.imread('/content/test.jpg')
im = resize_image(im)

#car parts
outputs2 = carparts_predictor(im)

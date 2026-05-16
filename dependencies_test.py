from facenet_pytorch import MTCNN
import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
import torchvision; print('torchvision:', torchvision.__version__)
import cv2; print('cv2:', cv2.__version__)
import numpy; print('numpy:', numpy.__version__)
import PIL; print('Pillow:', PIL.__version__)
from facenet_pytorch import MTCNN; print('facenet-pytorch: OK')

import onnxruntime; print('onnxruntime:', onnxruntime.__version__)
import sklearn; print('scikit-learn:', sklearn.__version__)
import tqdm; print('tqdm:', tqdm.__version__)
import onnx; print('onnx:', onnx.__version__)
mtcnn = MTCNN()
print("MTCNN loaded successfully")
print('opencv', cv2.__version__)

import insightface; print('insightface:', insightface.__version__)
print('--- ALL IMPORTS OK ---')

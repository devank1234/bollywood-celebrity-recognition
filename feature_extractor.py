#import os
#import pickle

#folder = 'data/Bollywood_celeb_face_localized'

#filenames = []

#for root, dirs, files in os.walk(folder):
 #   for file in files:
  #      if file.lower().endswith(('.jpg', '.jpeg', '.png')):
   #         filenames.append(os.path.join(root, file))

#pickle.dump(filenames, open('filenames.pkl', 'wb'))

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filenames.pkl','rb'))

model = VGGFace(
    model='resnet50',
    include_top=False,
    input_shape=(224,224,3),
    pooling='avg'
)

def feature_extractor(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img=np.expand_dims(img_array,axis=0)
    prepared_img=preprocess_input(expanded_img)

    result=model.predict(prepared_img).flatten()

    return result

features=[]

for file in tqdm(filenames):
    features.append(feature_extractor(file,model))

pickle.dump(features,open('embedding.pkl','wb'))


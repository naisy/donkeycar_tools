# Donkey Car 3
# python ~/projects/donkeycar/scripts/freeze_model.py --model=linear.h5 --output=linear.pb
# python /usr/lib/python3.6/dist-packages/uff/bin/convert_to_uff.py linear.pb
# python manage.py drive --model=linear.uff --type=tensorrt_linear

# Donkey Car 4
#https://github.com/onnx/tensorflow-onnx
#pip install -U tf2onnx
#python convert_h5_to_engine.py

########################################
# h5 to onnx
########################################
import tensorflow as tf
import onnx
import tf2onnx.convert
from tensorflow.keras.models import Model, load_model
model = load_model('linear.h5')
#model.save('linear.pb')

onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save(onnx_model, 'linear.onnx')


########################################
# onnx to engine (sh command)
########################################
import os
command = f'/usr/src/tensorrt/bin/trtexec --onnx=linear.onnx --saveEngine=linear.engine --fp16 --shapes=img_in:1x90x160x3 --explicitBatch'
os.system(command)


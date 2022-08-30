# donkeycar_tools
For Donkey Car linear model.

*   convert_h5_to_engine.py
*   convert_h5_to_tflite.py
*   convert_v2_to_v1.py
*   donkey-racer-bench.py
*   racer.py

### convert_h5_to_engine.py
Convert from Tensorflow keras model to TensorRT model.  
TensorRT model can use with racer.py.  
*   Requirement: linear.h5

### convert_h5_to_tflite.py
Convert from Tensorflow keras model to Tensorflow Lite model.  
Tensorflow Lite model can use with Donkey Car and racer.py.  
*   Requirement: linear.h5

### convert_v2_to_v1.py
Convert from Donkey Car 4.3 tub_v2 data to Donkey Car 3.x tub_v1 data.  
The future predictions made for donkeycar 3.x will be available.  
*   ~/mycar/data is original data
*   ~/mycar/data_v1 will be created

### donkey-racer-bench.py
Check the communication status with the Donkey Car Simulator.  

### racer.py
Simple racer.py for Donkey Car Simulator.  
The delay option is the sleep time after receiving the frame from the simulator until the control value is sent back to the server.  

When the donkey car saves the driving data of the simulator, the control value and the frame time are saved in a mismatched state.  
Since the data is saved with future-predicted values, you need to specify the delay option in simple racer.py to cancel this future-predicted value.  
When the simulator is a network server, the delay is close to 0.0 seconds. At the local server, around 0.2 secconds is good.  

Normal donkey car can run without problems because the timing of frame reception and control transmission is different during automatic driving as well as during recording. This delay option is mandatory for custom clients.

**youtube:**
#### Donkey Car Simulator TensorRT test
[![Donkey Car Simulator TensorRT test](http://img.youtube.com/vi/VpeiqdVh12g/default.jpg)](https://youtu.be/VpeiqdVh12g)

## Donkey Car OVERDRIVE3
*   make_slided_train_data.py
*   make_ai_to_train_data.py

### make_slided_train_data.py
Future prediction. Modify training data to improve movement.  
**usage:**
```
cd ~/mycar
wget https://raw.githubusercontent.com/naisy/donkeycar_tools/master/make_slided_train_data.py
python make_slided_train_data.py
```
*   ~/mycar/data is original data
*   ~/mycar/data_slided will be created

**youtube:**
#### Tamiya TT-02 DonkeyCar Drift Test2
[![Drift Test2](http://img.youtube.com/vi/iSLTgYGxONg/default.jpg)](https://www.youtube.com/watch?v=iSLTgYGxONg)

### make_ai_to_train_data.py
Donkey Car can save control values during autonomous driving by setting RECORD_DURING_AI in myconfig.py to True.
This is a conversion to use that value for training.  
**usage:**
```
cd ~/mycar
wget https://raw.githubusercontent.com/naisy/donkeycar_tools/master/make_ai_to_train_data.py
python make_ai_to_train_data.py
```
*   ~/mycar/data is original data
*   ~/mycar/data_ai will be created



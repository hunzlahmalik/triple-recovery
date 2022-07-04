# triple-recovery
A paper implementation in numpy python

# Requirements
- Python >= 3.7

# Installation
```bash
pip3 install triplerecovery
```
## Auto Installed
-  numpy >=1.22.4
-  opencv-python >=4.6.0.66

# Usage
## Embedding
```py
import cv2
import triplerecovery as trir

# image can be rgb or grey
# grey must have two dimetions so add cv2.
# if you know the image is grey then use cv2.IMREAD_GRAYSCALE

imarr=cv2.imread("<your image path>")
embeded_image=trir.embed(imarr).imarr
```
## Recovery
```py
import cv2
import triplerecovery as trir

imarr=cv2.imread("<your embeded image path>")
recovered_image=trir.recover(imarr).imarr
# OR for changed interpolation
recovered_image=trir.recover(imarr, interpolation = cv2.INTER_CUBIC).imarr
```

## Material
The images material and tests are available on this link: https://drive.google.com/drive/folders/1kamSYESP3HOLn9pyPLIR_IH1qB-aPhI6?usp=sharing
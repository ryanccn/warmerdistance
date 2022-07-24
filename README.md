# WarmerDistance

> Social distancing maintenance system based on computer vision

With the alarming spread of COVID-19 worldwide, we noticed the importance of social distancing for public health—One meter or more reduces the risk of infection by 82%. Inspired by an art installation, we decided to use ground projection as a reminder of human distancing, most importantly, putting forward the calculation of interpersonal distance with PyTorch and YOLOv5, which has already successfully run in several basic scenarios. Bubble space, which refers to one’s safe social distance in certain areas and will be actually projected on the ground, is the innovative concept we’d like to introduce. When the system detects another in one’s bubble space, the two projections of bubbles will squeeze and turn red to gently warn them. The need of public health safety, artistry, aesthetics, and futurism can all be satisfied by this system. And that’s how we picture ideal public space in the pandemic—distance is always needed, but warmer.

```bash
# Install dependencies
$ pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
$ pip install -r requirements.txt

# Run the script (requires network connection)
$ python main.py image input.png output.png
$ python main.py video input.mp4 output.mp4
$ python main.py live
```

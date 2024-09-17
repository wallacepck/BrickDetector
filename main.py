import logging
import sys
import base64
import time
import os
from os import listdir
from os.path import isfile, join, splitext
from timeit import default_timer
import argparse

import json
import cv2
from tkinter import Tk, messagebox
import eel
import numpy as np

from camera import VideoCamera

import brickenator
from brickenator import labelBricksWithBBox

parser = argparse.ArgumentParser(description='Assignment 1 main program, includes option for either Approach A and Approach B.')
parser.add_argument("--approach", required=True, type=str, help="Approach to use, either 'A' or 'B' ")


COLORS_RGB = {
  'GREY' : (122, 114, 112),
  'BLACK' : (61, 57, 58),
  'BLUE' : (89, 105, 138),
  'CYAN' : (67, 113, 136),
  'ORANGE' : (176, 80, 42),
  'YELLOW' : (196, 159, 70),
  'LIME' : (124, 128, 52),
  'GREEN' : (59, 94, 61)
}

EMT_RECOGNISED_NAME = {
  'GREY' : "Light Gray",
  'BLACK' : "Grey",
  'BLUE' : "Medium Blue",
  'CYAN' : "Medium Azure",

  '4' : "2x2",
  '6' : "2x3",
  '8' : "2x4",
  '16' : "2x8",
  '24' : "4x6"
}

VIDEO_FOLDER_PATH = "./web/video"
VIDEOS = {}

GROUPS = {"brick1" : "level1Collapse", "brick3" : "level2Collapse", "brick6" : "level3Collapse"}

# Read Images
empty = cv2.imread("./web/image/empty.png",cv2.IMREAD_GRAYSCALE)

# Setup the images to display in html file
@eel.expose
def setup():
  img_send_to_js(np.zeros_like(empty), "videoPlayerOutput")
  eel.redTime(False)

  for name, permutations in brickenator.BRICK_PERMUTATIONS.items():
    try:
      externalName = EMT_RECOGNISED_NAME[name]
    except KeyError:
      externalName = name.title()
    
    try:
      colorHex = '#%02x%02x%02x' % COLORS_RGB[name]

      for p in permutations:
        px = EMT_RECOGNISED_NAME[str(p)]
        ps = px.split('x')
        eel.addDetCategory(colorHex, " ".join([externalName, px]), name + str(p), ps[0], ps[1])
    except KeyError:
      logging.warning('{}\n{}'.format("Color has no color or permutations", name))

  files = [f for f in listdir(VIDEO_FOLDER_PATH) if f.endswith(".mp4")]

  for f in files:
    f_name = splitext(f)[0]
    eel.loadingBarUpdate(f)
    v = cv2.VideoCapture(join(VIDEO_FOLDER_PATH, f))

    ret, img = v.read()
    if not ret:
      logging.error('{}\n{}'.format("Could not read video file", f))
      continue

    downImg = cv2.resize(img, None, fx=0.13, fy=0.13, interpolation=cv2.INTER_NEAREST)

    ret, jpeg = cv2.imencode(".jpg", downImg)
    jpeg.tobytes()
    blob = base64.b64encode(jpeg) 
    blob = blob.decode("utf-8")

    try:
      group = GROUPS[f_name.split("_")[0]]
    except (KeyError, IndexError) as e:
      logging.error('{}\n{}'.format("Could not resolve video group", e.args))
      continue

    eel.addVideoToList(group, f_name, blob)
    VIDEOS[f_name] = f

  if brickDetector.name() == "Approach B":
    text_send_to_js("Brick Detector YOLO", "titleBoard")
  
  eel.loadingBarUpdate("demo.json")
  with open('web/demo.json') as f:
    data = json.load(f)[brickDetector.name()]
    for stage, metadata in data.items():
      eel.addTabToDemoPanel(stage, metadata)

  eel.loadingBarUpdate()
  
  
@eel.expose
def videoSelect(title):
  global currentTitle
  if title == currentTitle: #Ignore pylint warning here, it's just confused.
    return

  currentTitle = title
  text_send_to_js(VIDEOS[title], "videoPlayerTitle")
  start_video_feed(join(VIDEO_FOLDER_PATH, VIDEOS[title]))

#  Your code depend on image processing
# This is a sample code to change 
# and send processed image to JavaScript  
@eel.expose
def start_video_feed(src):
  global capture, global_generator_id, hook, brickDetector
  if src:
    capture = VideoCamera(src)  # Do not worry about the old capture resource, it'll be released by cv2 internally.
    eel.showPauseButton(True)
    global_generator_id += 1
    y = process(global_generator_id)
    for frame in y:
      img_send_to_js(frame, "videoPlayerOutput")

@eel.expose
def hookDemo(val):
  global hook
  hook = val

# Get capture from video feed
# Add ur codes to process here
def process(gen_id):
  frameCount = capture.video.get(cv2.CAP_PROP_FRAME_COUNT)
  eel.seekBarMax(frameCount-1)
  eel.seekBarValue(0)

  frameIndex = 0
  elapsed = 0
  last_active = None
  while True:
    # Exit if main generator is no longer this generator
    if gen_id != global_generator_id:
      return None

    # GUI check
    if eel.isGUISeeking()():
      frameIndex = int(eel.seekBarValue()())
      capture.setPosition(frameIndex)
      eel.acknowledgeSeekClick()

    success, frame = capture.get_frame()
    if success:
      elapsed = int(capture.video.get(cv2.CAP_PROP_POS_MSEC)//1000)
      text_send_to_js(f'{elapsed//60:02}:{elapsed%60:02}', "elapsed")

    start = default_timer()
    bricks, out, debug = brickDetector(frame, hook)
    end = default_timer()

    out = labelBricksWithBBox(out, bricks, EMT_RECOGNISED_NAME)

    if debug is not None:
      out = debug
      
    if end - start > 1/10:
      eel.redTime(True)
    else:
      eel.redTime(False)

    # GUI check
    if success and not eel.isGUISeeking()():
      frameIndex += 1
      eel.seekBarValue(frameIndex)
      eel.showPauseButton(True)
    else:
      eel.showPauseButton(False)

    active = {}
    for b in bricks:
      name = b.color + str(b.circleCount)
      if name in active:
        active[name] += 1
      else:
        active[name] = 1
      pass

    if last_active != active:
      for name, permutations in brickenator.BRICK_PERMUTATIONS.items():
        for p in permutations:
          qualifiedName = name + str(p)
          if qualifiedName in active:
            text_send_to_js(active[qualifiedName], qualifiedName)
            eel.setTransparency(1, qualifiedName + "_")
          else:
            text_send_to_js(0, qualifiedName)
            eel.setTransparency(0.1, qualifiedName + "_")

    yield out

# Stop Video Caturing
# Do not touch
@eel.expose
def stop_video_feed():
  capture.setActive(False)
  eel.showPauseButton(False)

@eel.expose
def toggle_video_feed():
  state = capture.toggle()
  if state:
    eel.seek(False) # Failsafe if the seek is locked up due to terrible HTML
  eel.showPauseButton(state)
  
# Send text from python to Javascript 
# Do not touch
def text_send_to_js(val,id):
  eel.updateTextSrc(val,id)()

# Send image from python to Javascript 
# Do not touch
def img_send_to_js(img, id):
  if np.shape(img) == () :
    
    eel.updateImageSrc("", id)()
  else:
    ret, jpeg = cv2.imencode(".jpg",img)
    jpeg.tobytes()
    blob = base64.b64encode(jpeg) 
    blob = blob.decode("utf-8")
    eel.updateImageSrc(blob, id)()

# Start function for app
# Do not touch
def start_app():
  try:
    start_html_page = 'index.html'
    eel.init('web')
    logging.info("App Started")

    eel.start('index.html', size=(1000, 800))
  except Exception as e:
    err_msg = 'Could not launch a local server'
    logging.error('{}\n{}'.format(err_msg, e.args))
    show_error(title='Failed to initialise server', msg=err_msg)
    logging.info('Closing App')
    sys.exit(1)

if __name__ == "__main__":
  capture = None
  hook = None
  global_generator_id = 0
  currentTitle = None

  args = parser.parse_args()
  if args.approach == "B":
    try:
      from deep import YoloBrickDetector
      brickDetector = YoloBrickDetector("yolov5s_bricks.pt")
    except ImportError as e:
      logging.error("Failed to import YOLOv5. Try setting up a new environment from environment_deep.yml.")
      raise e #reraise the error
  elif args.approach == "A":
    from brickenator import BrickenatorDetector
    brickDetector = BrickenatorDetector()
  else:
    raise ValueError(f'Expected "A" or "B" in argument "approach", got "{args.approach}"')

  start_app()
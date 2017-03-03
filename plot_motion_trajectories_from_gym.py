from PIL import Image, ImageDraw
import numpy as np
import gym

env = gym.make('Walker2d-v1')
height = 1000
width = 1000

s_height = 700
s_width = 700

step_len = 20
frame_skip = 4

canvas = Image.new('RGBA', (step_len * s_width, s_height), (0,0,0,255))

for i_iter in range(frame_skip * step_len):

  env.render()

  action = env.action_space.sample()
  env.step(action)

  if i_iter % frame_skip != 0: continue

  i, _, _ = env.viewer.get_image()

  from IPython import embed; embed()

  i = np.fromstring(i, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
  i = i[150:850, 150:850, :]

  img = Image.fromarray(i)
  imga = img.convert("RGBA")

  alpha = int(255.0 * i_iter / (frame_skip * step_len))

  pixeldata = list(imga.getdata())
  for i, pixel in enumerate(pixeldata):
    if pixeldata[i][0] + pixeldata[i][0] + pixeldata[i][0] == 0:
      pixeldata[i] = (0, 0, 0, 0)
    else:
      pixeldata[i] = (pixeldata[i][0], pixeldata[i][1], pixeldata[i][2], alpha)
  imga.putdata(pixeldata)

  offset = (i_iter/frame_skip * s_width/5, 0)
  canvas.paste(imga, offset, imga)


canvas.save("motion.png")

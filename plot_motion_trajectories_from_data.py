from PIL import Image, ImageDraw
import numpy as np
import gym
import cPickle as pickle

env = gym.make('Walker2d-v1')
height = 1000
width = 1000

s_height = 700
s_width = 700

step_len = 20
frame_skip = 50

k = 9

data = pickle.load(open('trajectories_Walker2dDynamic-v1_0_15', 'r'))

canvas = Image.new('RGBA', (step_len * s_width, s_height), (0,0,0,255))

for i_iter in range(frame_skip * step_len):

  env.render()

  # dimensions
  # 0: qpos, qvel, reward
  # 1: num. rollout = 100
  # 2: num. iter

  if i_iter >= len(data[0][k]): break

  if i_iter >= 950: break

  qpos = data[0][k][i_iter]
  qvel = data[1][k][i_iter]

  # from IPython import embed; embed()
  env.set_state(qpos, qvel)
  # action = env.action_space.sample()
  # env.step(action)

  if i_iter % frame_skip != 0: continue
  # if i_iter >= 750: break

  print(i_iter)
  i, _, _ = env.viewer.get_image()

  i = np.fromstring(i, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
  i = i[150:850, 150:850, :]

  img = Image.fromarray(i)
  imga = img.convert("RGBA")

  alpha = int(200.0 * i_iter / min(len(data[0][k]), frame_skip * step_len, 950) + 55.0)

  pixeldata = list(imga.getdata())
  for i, pixel in enumerate(pixeldata):
    if pixeldata[i][0] + pixeldata[i][0] + pixeldata[i][0] == 0:
      pixeldata[i] = (0, 0, 0, 0)
    else:
      pixeldata[i] = (pixeldata[i][0], pixeldata[i][1], pixeldata[i][2], alpha)
  imga.putdata(pixeldata)

  offset = (i_iter/frame_skip * s_width/4, 0)
  canvas.paste(imga, offset, imga)

canvas.save("motion.png")

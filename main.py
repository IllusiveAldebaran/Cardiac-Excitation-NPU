#!/usr/bin/python3
# Code to simulate Aliev-Panfilov
# 11/28/2025

import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

# constants, original float64 simulation values commented out
a = 0.155 # 0.1
b = 0.155 # 0.1
kk = 8.0  # 8.0
M1 = 0.07 # 0.07
M2 = 0.3  # 0.3
epsilon = 0.01
d = 5e-5  #e-5


def main():
  argparser = argparse.ArgumentParser(
          prog="Aliev-Panflov 2d simulation",
          description="Simulates Cardiac Electrophysiology Simulation"
          )
  argparser.add_argument("-n", type=int, default=32)
  argparser.add_argument("-i", type=int, default=100)
  argparser.add_argument("-p", type=int, default=1)
  args = argparser.parse_args()
  niters = args.i
  n = args.n
  m = n
  plot_freq = args.p

  E       = torch.zeros((m+2), (n+2), dtype=torch.bfloat16);
  E_prev  = torch.zeros((m+2), (n+2), dtype=torch.bfloat16);
  R       = torch.zeros((m+2), (n+2), dtype=torch.bfloat16);

  dx = 1.0/(n-1);
  rp= kk*(b+1)*(b+1)/4;
  dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
  dtr=1/(epsilon+((M1/M2)*rp));
  ddt = 0.95*dte if (dte<dtr) else 0.95*dtr;
  dt = ddt;
  alpha = d*dt/(dx*dx);


  id_x = 0
  id_y = 0
  # generic (or average) tile width and height
  gtw = n #//px
  gth = m #//py
  # tile width and tile height
  ptw = gtw #+ (id_x == cb.px - 1)*(cb.n % cb.px)
  pth = gth #+ (id_y == cb.py - 1)*(cb.m % cb.py)
  # top left index a tile
  tlt = 0;
  row_length = ptw + 2; # use this when iterating down
  px = 1
  py = 1


  # I don't know why this chooses to go from range [1,m+1). 
  # I notice in my previous code I did [1,m+2), but going back to the first version
  # of the problem it went to m+1, so that's what I'm doing here
  for i in range(1, m+1):
    for j in range(n+2):
      #// Need to compute (n+1)/2 rather than n/2 to work with odd numbers
      if (j == 0 or j == (n+1) or j < ((n+1)//2+1) ):
          continue
      else:
          E_prev[i][j] = 1.0;
  
  for i in range(m+2):
    for j in range(n+2):
      if (j == 0 or j == (n+1) or i < ((m+1)//2+1)-1):
          #print(rowIndex, ((m+1)/2+1)-1)
        continue
      else:
          R[i][j] = 1.0



  # Create a colormap with linear interpolation
  cmap = LinearSegmentedColormap.from_list(
          "blue_white_red",
          [
              (0.0, "blue"),   # value = 0
              (0.75, "white"), # value = 0.75
              (1.0, "red"),    # value = 1
              ]
          )

  plt.ion()  # turn on interactive mode
  fig, ax = plt.subplots()

  # Initial image
  img = ax.imshow(E.reshape(m+2, n+2).float().cpu().numpy(),
                  cmap=cmap, interpolation='nearest', extent=[0, n+2, 0, m+2])
  fig.colorbar(img, ax=ax, ticks=[0, 0.75, 1])
  img.set_clim(0, 1)


  # move to cuda device if available
  if torch.cuda.is_available():
    print("Cuda device found. Moving Meshes to device.")
    cuda_dev = torch.device("cuda")
    E_prev.to(cuda_dev)
    E.to(cuda_dev)
    R.to(cuda_dev)

  for niter in range(niters):

    if (id_y == 0):
      #print('Processor %d has top side edges to fill\n', myrank);
      E_prev[-1, 1:-1] = E_prev[-3, 1:-1]

    if (id_y == py - 1):
      #printf("Processor %d has bottom side edges to fill\n", myrank);
      E_prev[0, 1:-1] = E_prev[2, 1:-1]

    if (id_x == px - 1):
      #printf("Processor %d has right side edges to fill\n", myrank);
      E_prev[1:-1, -1] = E_prev[1:-1, -3]

    if (id_x == 0):
      #printf("Processor %d has left side edges to fill\n", myrank);
      E_prev[1:-1, 0] = E_prev[1:-1, 2]


    E_int = E[1:-1, 1:-1]
    E_prev_int = E_prev[1:-1, 1:-1]
    R_int = R[1:-1, 1:-1]

    # Solve the ODE, advancing excitation and recovery variables
    E_int[:] = -dt*(kk*E_prev_int[:]*(E_prev_int[:]-a)*(E_prev_int[:]-1)+E_prev_int[:]*R_int[:]);
    R_int[:] += dt*(epsilon+M1* R_int[:]/( E_prev_int[:]+M2))*(-R_int[:]-kk*E_prev_int[:]*(E_prev_int[:]-b-1));

    # PDE update (diffusion term / Laplacian)
    laplacian = (
              E_prev[2:  , 1:-1]
            + E_prev[ :-2, 1:-1]
            + E_prev[1:-1,  :-2]
            + E_prev[1:-1, 2:  ]
            - 4 * E_prev[1:-1, 1:-1]  
        )

    E_int[:] += E_prev_int + alpha * laplacian

    # suggestion from chatgpt is convolution:
    # Not implementing or trying until after NPU is targetted. Then we can play around
    #kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=E.dtype, device=E.device)
    #laplacian = torch.nn.functional.conv2d(E_prev.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)



#///////////////   MAIN KERNEL END   //////////////////////////////////////////

    # Swap current and previous meshes
    E_prev, E = E, E_prev

    if (niter % plot_freq == 0):
      img.set_data(E.float().cpu().numpy())
      ax.set_title(f"Excitation Mesh iter={niter}")
      fig.canvas.draw()
      fig.canvas.flush_events()

  # end of 'niter' loop at the beginning



if __name__ == "__main__":
    main()

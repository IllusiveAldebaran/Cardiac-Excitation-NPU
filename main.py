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

# constants
a = 0.1
b = 0.1
kk = 8.0
M1 = 0.07
M2 = 0.3
epsilon = 0.01
d = 5e-5


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
  bounds = [0, 0.5]  # boundaries between the categories

  plt.ion()  # turn on interactive mode
  fig, ax = plt.subplots()

  # Initial image
  img = ax.imshow(E.reshape(m+2, n+2).float().cpu().numpy(),
                  cmap=cmap, interpolation='nearest', extent=[0, n+2, 0, m+2])
  fig.colorbar(img, ax=ax, ticks=[0, 0.75, 1])
  img.set_clim(0, 1)

  for niter in range(niters):

    if (id_y == 0):
      #print('Processor %d has top side edges to fill\n', myrank);
      E_prev[0, 1:-1] = E_prev[-2, 1:-1]

    if (id_y == py - 1):
      #printf("Processor %d has bottom side edges to fill\n", myrank);
      E_prev[-1, 1:-1] = E_prev[-2, 1:-1]

    if (id_x == px - 1):
      #printf("Processor %d has right side edges to fill\n", myrank);
      E_prev[1:-1, -1] = E_prev[1:-1, -2]

    if (id_x == 0):
      #printf("Processor %d has left side edges to fill\n", myrank);
      E_prev[1:-1, 0] = E_prev[1:-1, 2]

    # Solve the ODE, advancing excitation and recovery variables
    for i in range(1, ptw+1):
      for j in range(1, pth+1):
        E[i][j] = -dt*(kk*E_prev[i][j]*(E_prev[i][j]-a)*(E_prev[i][j]-1)+E_prev[i][j]*R[i][j]);
        R[i][j] += dt*(epsilon+M1* R[i][j]/( E_prev[i][j]+M2))*(-R[i][j]-kk*E_prev[i][j]*(E_prev[i][j]-b-1));

        # This is PDE case. It may be worth again using an unfused loop because of the adjacent memory access patterns
        E[i][j] += E_prev[i][j]+alpha*(E_prev[i][j+1]+E_prev[i][j-1]-4*E_prev[i][j]+E_prev[i+1][j]+E_prev[i-1][j])

    # Solve for the excitation, a PDE edge cases
    #for i in range(1, ptw+1):
    #  for j in range(1, pth+1):
    #    E[i][j] += E_prev[i][j]+alpha*(E_prev[i][j+1]+E_prev[i][j-1]-4*E_prev[i][j]+E_prev[i+1][j]+E_prev[i-1][j])


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

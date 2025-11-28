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

  E       = torch.zeros((m+2)*(n+2), dtype=torch.bfloat16);
  E_prev  = torch.zeros((m+2)*(n+2), dtype=torch.bfloat16);
  R       = torch.zeros((m+2)*(n+2), dtype=torch.bfloat16);

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


  for i in range(0, (m+1)*(n+2)):
    E_prev[i] = R[i] = 0;
  
  for i in range(n+2, (m+2)*(n+2)):
    colIndex = i % (n+2) #		// gives the base index (first row's) of the current index
  
    #// Need to compute (n+1)/2 rather than n/2 to work with odd numbers
    if (colIndex == 0 or colIndex == (n+1) or colIndex < ((n+1)/2+1) ):
        continue
    else:
        E_prev[i] = 1.0;
  
  
  
  for i in range(0, (m+2)*(n+2)):
    rowIndex = i // (n+2);	# gives the current row number in 2D array representation
    colIndex = i % (n+2);		# gives the base index (first row's) of the current index
  
    # Need to compute (m+1)/2 rather than m/2 to work with odd numbers
    if (colIndex == 0 or colIndex == (n+1) or rowIndex < ((m+1)/2+1)-1):
        #print(rowIndex, ((m+1)/2+1)-1)
      continue
    else:
        R[i] = 1.0



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

    # Solve the ODE, advancing excitation and recovery variables
    for j in range(1, pth+1):
      tile_row   = (j)*(row_length);
      # offset tile_row
      for i in range(1, ptw+1):
        E[i+tile_row] = -dt*(kk*E_prev[i+tile_row]*(E_prev[i+tile_row]-a)*(E_prev[i+tile_row]-1)+E_prev[i+tile_row]*R[i+tile_row]);
        R[i+tile_row] += dt*(epsilon+M1* R[i+tile_row]/( E_prev[i+tile_row]+M2))*(-R[i+tile_row]-kk*E_prev[i+tile_row]*(E_prev[i+tile_row]-b-1));


    # Solve for the excitation, a PDE edge cases, with THE BORDERS
    for j in range(2, pth):
      tile_row = j*(row_length); # iterative row offset
      # offset tile_row
      for i in range(2, ptw):
        E[i+tile_row] += E_prev[i+tile_row]+alpha*(E_prev[i+tile_row+1]+E_prev[i+tile_row-1]-4*E_prev[i+tile_row]+E_prev[i+tile_row + row_length]+E_prev[i+tile_row - row_length])


    if (id_y == 0):
      #print('Processor %d has top side edges to fill\n', myrank);
      edge = 1; # get top ghost cells
      for i in range(ptw):
        E_prev[i+edge] = E_prev[i+edge + (row_length)*2]


    if (id_y == py - 1):
      #printf("Processor %d has bottom side edges to fill\n", myrank);
      edge = (pth + 1) * (row_length) + 1
      for i in range(ptw):
        E_prev[i+edge] = E_prev[i+edge - (row_length)*2]


    if (id_x == px - 1):
      #printf("Processor %d has right side edges to fill\n", myrank);
      edge = row_length + ptw + 1
      for j in range(pth):
        E_prev[j*(row_length)+edge] = E_prev[j*(row_length) - 2+edge]

    if (id_x == 0) :
      #printf("Processor %d has left side edges to fill\n", myrank);
      edge = row_length
      for j in range(pth):
        E_prev[j*(row_length)+edge] = E_prev[j*(row_length) + 2+edge]


    ## Solve for the excitation, a PDE edge cases
    ## just around the tile though

    ## top (row_length offset)
    for i in range(1, ptw+1):
      E[i+row_length] += E_prev[i+row_length]+alpha*(E_prev[i+1+row_length]+E_prev[i-1+row_length]-4*E_prev[i+row_length]+E_prev[i + row_length+row_length]+E_prev[i - row_length+row_length]);

    #  bottom (row_length offset)
    for i in range(1, ptw+1):
      E[i+pth*row_length] += E_prev[i+pth*row_length]+alpha*(E_prev[i+1+pth*row_length]+E_prev[i-1+pth*row_length]-4*E_prev[i+pth*row_length]+E_prev[i + row_length+pth*row_length]+E_prev[i - row_length+pth*row_length]);
    #for (j = 2; j < pth; j++) { 

    # sides
    for j in range(2, pth):
      tile_row = j*(row_length); # // iterative row offset
      # tilw_row offset
      E[1+tile_row] += E_prev[1+tile_row]+alpha*(E_prev[1+1+tile_row]+E_prev[1-1+tile_row]-4*E_prev[1+tile_row]+E_prev[1 + row_length+tile_row]+E_prev[1 - row_length+tile_row]);
      E[ptw+tile_row] += E_prev[ptw+tile_row]+alpha*(E_prev[ptw+1+tile_row]+E_prev[ptw-1+tile_row]-4*E_prev[ptw+tile_row]+E_prev[ptw + row_length+tile_row]+E_prev[ptw - row_length+tile_row]);

#///////////////   MAIN KERNEL END   //////////////////////////////////////////

    # Swap current and previous meshes
    E_prev, E = E, E_prev

    if (niter % plot_freq == 0):
      img.set_data(E.reshape(m+2, n+2).float().cpu().numpy())
      ax.set_title(f"Excitation Mesh iter={niter}")
      fig.canvas.draw()
      fig.canvas.flush_events()

  # end of 'niter' loop at the beginning



if __name__ == "__main__":
    main()

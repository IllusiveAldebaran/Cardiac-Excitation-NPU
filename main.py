#!/usr/bin/python3
# Code to simulate Aliev-Panfilov
# 11/28/2025

import os
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
    args = argparser.parse_args()
    n = args.n
    m = n

    #E = alloc1D(cb.m+2,cb.n+2);
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


    #for (i=0; i < (m+1)*(n+2); i++)
    for i in range(0, (m+1)*(n+2)):
      E_prev[i] = R[i] = 0;

    #for (i = (n+2); i < (m+2)*(n+2); i++):
    for i in range(n+2, (m+2)*(n+2)):
      colIndex = i % (n+2) #		// gives the base index (first row's) of the current index

      #// Need to compute (n+1)/2 rather than n/2 to work with odd numbers
      if (colIndex == 0 or colIndex == (n+1) or colIndex < ((n+1)/2+1) ):
        continue

      E_prev[i] = 1.0;


    #for (i = 0; i < (m+2)*(n+2); i++) {
    for i in range(0, (m+2)*(n+2)):
      rowIndex = i // (n+2);	# gives the current row number in 2D array representation
      colIndex = i % (n+2);		# gives the base index (first row's) of the current index

      # Need to compute (m+1)/2 rather than m/2 to work with odd numbers
      if (rowIndex < ((m+1)/2+1)-1):
        #print(rowIndex, ((m+1)/2+1)-1)
        continue

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

    plt.imshow(R.reshape(m+2,n+2).float().cpu().numpy(), cmap=cmap, interpolation='nearest')
    plt.colorbar(ticks=[0, 0.75, 1])
    plt.title("Color-coded matrix")
    plt.show()
    plt.imshow(E_prev.reshape(m+2,n+2).float().cpu().numpy(), cmap=cmap, interpolation='nearest')
    plt.show()




if __name__ == "__main__":
    main()

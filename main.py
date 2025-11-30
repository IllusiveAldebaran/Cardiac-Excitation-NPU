#
# Code to simulate Aliev-Panfilov
# 11/28/2025

import os
import math
import numpy as np
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import time

# For NPU:
from ml_dtypes import bfloat16
import sys

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.utils.config import cxx_header_path


# constants, original float64 simulation values commented out
a = 0.155 # 0.1
b = 0.155 # 0.1
kk = 8.0  # 8.0
M1 = 0.07 # 0.07
M2 = 0.3  # 0.3
epsilon = 0.01
d = 5e-5  #e-5


# global variables
niters = n = m = plot_freq = 0
dx = rp = dte = dtr = ddt = dt = alpha = 0.0

class PlotState:
    fig = None
    ax  = None
    img = None
    cmap = None

P = PlotState() 

def init_meshes(E_prev, R):
  # I don't know why this chooses to go from range [1,m+1). 
  # I notice in my previous code I did [1,m+2), but going back to the first version
  # of the problem it went to m+1, so that's what I'm doing here
  # TODO Rewrite this much simler using range of indices
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

def init_plot(E_np):
  # Create colormap once
  # Create a colormap with linear interpolation
  cmap = LinearSegmentedColormap.from_list(
          "blue_white_red",
          [
              (0.0, "blue"),   # value = 0
              (0.75, "white"), # value = 0.75
              (1.0, "red"),    # value = 1
              ]
          )


  plt.ion()
  fig, ax = plt.subplots()

  img = ax.imshow(
      E_np,
      cmap=cmap,
      interpolation='nearest',
      extent=[0, n+2, 0, m+2]
  )

  fig.colorbar(img, ax=ax, ticks=[0.0, 0.75, 1.0])
  img.set_clim(0.0, 1.0)

  P.img, P.fig, P.ax = img, fig, ax
    

def update_plot(E, device, niter):
    P.img.set_data(E) # remember # must be np array
    P.ax.set_title(f"Excitation Mesh {device} iter={niter}")
    P.fig.canvas.draw()
    P.fig.canvas.flush_events()


def aliev_panfilov_reference(E_prev, R, E):
  id_x = 0
  id_y = 0
  # generic (or average) tile width and height
  gtw = n #//px
  gth = m #//py
  # tile width and tile height
  ptw = gtw #+ (id_x == cb.px - 1)*(cb.n % cb.px)
  pth = gth #+ (id_y == cb.py - 1)*(cb.m % cb.py)
  # top left index a tile
  px = 1
  py = 1

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

    if (plot_freq > 0 and niter % plot_freq == 0):
      update_plot(E.float().cpu().numpy(), "dev", niter)

    # Swap current and previous meshes
    if (niter != niters-1):
        E_prev, E = E, E_prev
#///////////////   MAIN KERNEL END   //////////////////////////////////////////

@iron.jit(is_placed=False)
def aliev_panfilov_npu(E_prev_iron, R_iron, E_iron):
    id_x = 0
    id_y = 0
    # generic (or average) tile width and height
    gtw = n #//px
    gth = m #//py
    # tile width and tile height
    ptw = gtw #+ (id_x == cb.px - 1)*(cb.n % cb.px)
    pth = gth #+ (id_y == cb.py - 1)*(cb.m % cb.py)
    # top left index a tile
    px = 1
    py = 1

    N = E_prev_iron.shape[0]  # Tensor size
    #print("N is: ", n)
    element_type = E_iron.dtype

    # --------------------------------------------------------------------------
    # In-Array Data Movement
    # --------------------------------------------------------------------------

    in_ty = np.ndarray[(N,N), np.dtype[element_type]]
    out_ty = np.ndarray[(N,N), np.dtype[element_type]]

    of_x = ObjectFifo(in_ty, name="x")
    of_y = ObjectFifo(in_ty, name="y")
    of_z = ObjectFifo(out_ty, name="z")

    # --------------------------------------------------------------------------
    # Task each core will run
    # --------------------------------------------------------------------------

    ode_kernel = ExternalFunction(
        "aliev_panfilov_kernel",
        source_file=os.path.join(os.path.dirname(__file__), "stencil_kernels.cc"),
        arg_types=[in_ty, in_ty, out_ty, np.int32, np.int32, np.float32, np.float32],
        include_dirs=[cxx_header_path()],
    )

    def core_body(of_x, of_y, of_z, aie_kernel_solver):
        elem_x = of_x.acquire(1)
        elem_y = of_y.acquire(1)
        elem_z = of_z.acquire(1)
        aie_kernel_solver(elem_x, elem_y, elem_z, niters, n+2, dt, alpha)
        of_x.release(1)
        of_y.release(1)
        of_z.release(1)
    

    worker = Worker(
        core_body, fn_args=[of_x.cons(), of_y.cons(), of_z.prod(), ode_kernel]
    )

    # --------------------------------------------------------------------------
    # DRAM-NPU data movement and work dispatch
    # --------------------------------------------------------------------------

    rt = Runtime()
    with rt.sequence(in_ty, in_ty, out_ty) as (a_x, a_y, c_z):
        rt.start(worker)
        rt.fill(of_x.prod(), a_x)
        rt.fill(of_y.prod(), a_y)
        rt.drain(of_z.cons(), c_z, wait=True)

    # --------------------------------------------------------------------------
    # Place and generate MLIR program
    # --------------------------------------------------------------------------

    my_program = Program(iron.get_current_device(), rt)
    return my_program.resolve_program(SequentialPlacer())


def main():
  argparser = argparse.ArgumentParser(
          prog="Aliev-Panflov 2d simulation",
          description="Simulates Cardiac Electrophysiology Simulation"
          )
  argparser.add_argument("-n", type=int, default=32)
  argparser.add_argument("-i", type=int, default=100)
  argparser.add_argument("-p", type=int, default=1)
  global niters, n, m, plot_freq, dx, rp, dte, dtr, ddt, dt, alpha
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


  init_meshes(E_prev, R)
  init_plot(E.reshape(m+2, n+2).float().cpu().numpy()) # conversion for array np

  # Define tensor shapes and data types
  element_type = bfloat16

  # Construct an input tensor and an output zeroed tensor
  # The two tensors are in memory accessible to the NPU
  E_prev_iron = iron.zeros(E.shape, dtype=element_type, device="npu")
  R_iron = iron.zeros(R.shape, dtype=element_type, device="npu")

  # copy values cause Tensor.__init__ has a bug apparently
  # have to do this dumn workaround now
  E_prev_iron[:] = E_prev.to(torch.float32).cpu().numpy()
                    #np.eye(E_prev.shape[0], dtype=element_type) # using identity matrices to debug 
  R_iron[:] = R.to(torch.float32).cpu().numpy() #np.eye(R.shape[0], dtype=element_type) # again using identity matrices to debug
                    #R.to(torch.float32).cpu().numpy()

  E_iron = iron.zeros(64,64, dtype=element_type, device="npu")

  # Main stuff
  aliev_panfilov_npu(E_prev_iron, R_iron, E_iron)

  # Display final npu sim
  if plot_freq > 0:
    update_plot(E_iron.numpy().astype(np.float32), "NPU", 0)


  aliev_panfilov_reference(E_prev, R, E);





  errors = 0
  close_errors = 0
  
  for i in range(m):
    for j in range(n):
      actual = E_iron[i][j]
      ref = E[i][j]

      if actual != ref:
        if math.isclose(float(actual), float(ref), rel_tol=1e-2, abs_tol=1e-2):
          close_errors += 1
        else:
          print(f"Error at ({i}, {j}): {actual} != {ref}")
          errors += 1
      else:
        print(f"Correct at ({i}, {j}): {actual} == {ref}")

  # If the result is correct, exit with a success code
  # Otherwise, exit with a failure code
  if not errors:
    print("\nPASS!\n")
    sys.exit(0)
  elif close_errors > 0:
    print(f"\nOver {close_errors} close error(s)!\n")
    sys.exit(0)
  else:
    print("\nError count: ", errors)
    print("\nfailed.\n")
    sys.exit(1)



if __name__ == "__main__":
    main()

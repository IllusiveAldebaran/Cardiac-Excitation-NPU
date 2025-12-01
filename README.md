# Introduction


A Cardiac Electrophysiology Simulation targetting the AMD NPU.
Code performs a 2D Stencil Simulation on two meshes. An `E`xcitation and `R`ecovery mesh are solved for in discrete measures of time with a PDE and ODE.

`48x48` sim over 1000 iterations done on a single NPU tile. (500 iterationsX2 because NPU kernels is currently run twice as `-i` states)
![NPU_pic_supposed_to_be_here](https://github.com/IllusiveAldebaran/Cardiac-Excitation-NPU/blob/master/media/n_48_i_500_npu.png)

`256x256` sim over 3500 iterations done on cpu.
![cpu_pic_supposed_to_be_here](https://github.com/IllusiveAldebaran/Cardiac-Excitation-NPU/blob/master/media/n_256_i_3500_cpu.png)
                               https://github.com/IllusiveAldebaran/Cardiac-Excitation-NPU/blob/master/media/n_256_i_3500_cpu.png

`256x256` sim over 5000 iterations on cpu.
![another_cpu_pic_supposed_to_be_here](https://github.com/IllusiveAldebaran/Cardiac-Excitation-NPU/blob/master/media/n_256_i_5000_cpu.png)
 
### NPU Yes or No?

As a stencil problem the NPU's datamemory sharing feature becomes very interesting. And the large vectorization available per core means that each tile can perform a smaller section of the simulation independently. The major downside is that the original code uses the `double` type, but the only floating type that the [NPU natively supports](https://docs.amd.com/r/en-US/am020-versal-aie-ml/Functional-Overview) is bfloat16, and can emulate fp32 using bfloat16.

As such, for this problem we use bfloat16.

*On a side node, the AIE1 natively supports fp32, so it would make it interesting to target but is only available on FPGAs and the use of IRON for them isn't very straightforward*

### Original Problem
 
This was first given as an assignment for learning MPI programming. Here's an old version of the assignment: [UCSD CSE260 PA3](https://cseweb.ucsd.edu/classes/fa12/cse260-b/HW/A3/). It is based on the paper: <u>[**A Simple Two-variable Model of Cardiac Excitation**](https://www.sciencedirect.com/science/article/pii/0960077995000895)</u>.


# Status
You'll need to follow the instructions to install xrt for [MLIR-AIE Repo](https://github.com/Xilinx/mlir-aie). 

To run a `256x256` sim on CPU for 3500 iterations with plotting every 250 iterations:
```
python3 main.py -n 256 -i 3500 -p 250
```

If `cuda.is_avail()` returns true it will try to target that device instead of cpu.

Running CPU/GPU is the reference. Comment out `aliev_panfilov_npu()` or `aliev_panfilov_reference()` depending on which one you wish to run. Not an argument yet. Right now it is set to run npu first, then the reference. However the npu is more limited by tile memory space. Running with a dimension `n` greater than 64 for more than 400 iterations will probably simply throw an error.

After commenting out the reference to run the npu kernel:
```
python3 main.py -n 48 -i 500 -p 1000
```

*note that -p plotting is irrelevant as the npu version currently only displays when it is done iterating. And again, this will run 1000 times because the current IRON kernel calls the AIE C API kernel twice*


As a current status of this project, some of the next things I wish to be done are:
  * Target multiple tiles + exchanging cell walls
  * Return output status every few iterations from the NPU to plot better
  * Make use of the memory tile
  * Figure out how to reduce the large memory usage
  * Allow animation and to be saved + dissaggregate plotting into a fully seperate class
  * Test GPU and iGPU (need ROCm)
  * Optimize kernel in C API with vectorization
  * Figure out how to use fp32 in the NPU (despite it only being emulated)


### Extra Notes and Resources

If you'd like to read more on what an AI-Engine is, AIE1 vs AIE-ML vs AIEv2, or how to program for this architecture, you can read this first page made by myself and teammate for a high level overview:
[Making AIE dev easier](https://making-aie-dev-easier-readthedocs.readthedocs.io/en/latest/setup_and_explanation.html)

It is also a guide on how to using the Vitis C++ Graph Code and AIE Kernels, an alternate method of programming the AI Engines.

For development please read the [IRON Repo](https://github.com/amd/IRON). It's relatively new though. More fleshed out is the [MLIR-AIE Repo](https://github.com/Xilinx/mlir-aie).


[Guided Graph](https://docs.amd.com/p/ai-engine-development)

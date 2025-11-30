#include <aie_api/aie.hpp>
#include <aie_api/vector.hpp>
#include <aie_kernels/aie_kernel_utils.h>

extern "C" {

float a = 0.155;
float b = 0.155;
float kk = 8.0;
float M1 = 0.07;
float M2 = 0.3;
float epsilon = 0.01;
float d = 5e-5;

// Simulation has a few variables that are a bit unecessary at the moment
// For example, n=64 is the only thing that works.
void aliev_panfilov_kernel(bfloat16 *__restrict A, bfloat16 *__restrict B, bfloat16 *__restrict C,
		int niters, 
		int n, 
		float dt,
		float alpha
		) {
  event0();
  bfloat16* E_prev = A;
  bfloat16* R = B;
  bfloat16* E = C;

  // niter loop
  for(int niter = 0; niter < niters; ++niter){

    // calculate ghost cells and edges first...
    for(int i = 1; i < n-1; ++i){
	E_prev[i]=E_prev[i+2*n]; // top
	E_prev[(n-1)*n+i]=E_prev[(n-3)*n+i]; // bottom
	E_prev[(i+1)*n-1]=E_prev[(i+1)*n-3]; // right
	E_prev[i*n]=E_prev[i*n+2]; // left
    }

    // Look at all interior cells [1,n-1)[1,n-1) (0 based indexing obv)
    for(int i = 1; i < n-1; ++i){
      for(int j = 1; j < n-1; ++j){
          E[i*n+j] = -dt*(kk*E_prev[i*n+j]*(E_prev[i*n+j]-a)*(E_prev[i*n+j]-1)+E_prev[i*n+j]*R[i*n+j]);
          R[i*n+j] += dt*(epsilon+M1* R[i*n+j]/( E_prev[i*n+j]+M2))*(-R[i*n+j]-kk*E_prev[i*n+j]*(E_prev[i*n+j]-b-1));
          E[i*n+j] += E_prev[i*n+j]+alpha*(E_prev[i*n+j+1]+E_prev[i*n+j-1]-4*E_prev[i*n+j]+E_prev[i*n+j + (n)]+E_prev[i*n+j - (n)]);
      }
    }

    // swap pointers
    bfloat16* tmp = E_prev;
    E_prev = E;
    E = tmp;
  }
  event1();

}
}

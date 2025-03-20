import cupy as cp
import math

class HPDFT_CuPy_Shared_Double_Optimized:
    """
    Fully Optimized HPDFT Implementation in CuPy Using Shared Memory (Double Precision)

    This kernel now:
      1. Loads input from global memory into shared memory.
      2. Performs a bit-reversal permutation using __brev to reorder the data.
      3. Precomputes twiddle factors once per FFT stage in shared memory (using sincos).
      4. Executes iterative FFT butterfly stages using fused multiply-add (fma).
      5. Scales the result for the inverse FFT and writes back to global memory.
    
    The algorithm works with cp.complex128 arrays for sizes N that are powers of 2,
    provided that N fits within one CUDA block (e.g., N <= 1024).
    """
    def __init__(self):
        kernel_code = r'''
        #include <cuComplex.h>
        #include <math.h>
        
        extern "C"
        __global__ void fft_shared(const cuDoubleComplex * __restrict__ in,
                                   cuDoubleComplex * __restrict__ out,
                                   int N, int inverse)
        {
            // Allocate dynamic shared memory:
            // - First N elements for FFT data.
            // - Next N/2 elements for twiddle factors.
            extern __shared__ char shared_mem[];
            cuDoubleComplex *s_data = (cuDoubleComplex*) shared_mem;
            cuDoubleComplex *twiddle = (cuDoubleComplex*)(shared_mem + N * sizeof(cuDoubleComplex));
            
            int tid = threadIdx.x;
            
            // Load input from global memory into shared memory.
            if (tid < N)
                s_data[tid] = in[tid];
            __syncthreads();
            
            // Compute number of stages: stages = log2(N)
            int stages = 0, temp = N;
            while (temp > 1) {
                stages++;
                temp >>= 1;
            }
            
            // --- Bit-Reversal Permutation ---
            // Compute the bit-reversed index using __brev and shift.
            unsigned int rev = __brev(tid) >> (32 - stages);
            __syncthreads();
            if (tid < rev) {
                cuDoubleComplex tmp = s_data[tid];
                s_data[tid] = s_data[rev];
                s_data[rev] = tmp;
            }
            __syncthreads();
            
            // FFT iterative butterfly computation.
            const double PI = 3.14159265358979323846;
            int sign = (inverse ? 1 : -1);
            for (int s = 1; s <= stages; s++) {
                int m = 1 << s;         // m = 2^s (size of current sub-FFT)
                int half_m = m >> 1;
                double angle_unit = sign * (2.0 * PI / m);
                
                // Precompute twiddle factors for this stage once.
                if (tid < half_m) {
                    double theta = angle_unit * tid;
                    // sincos computes sine and cosine in one call.
                    sincos(theta, &twiddle[tid].y, &twiddle[tid].x);
                }
                __syncthreads();
                
                // Determine the position within the current sub-FFT block.
                int j = tid % m;
                int base = tid - j;
                if (j < half_m) {
                    cuDoubleComplex u = s_data[base + j];
                    cuDoubleComplex v = s_data[base + j + half_m];
                    cuDoubleComplex w = twiddle[j];
                    
                    cuDoubleComplex t_val;
                    // Compute the product v * w using fused multiply-add.
                    t_val.x = fma(v.x, w.x, -v.y * w.y);
                    t_val.y = fma(v.x, w.y,  v.y * w.x);
                    
                    // Butterfly update.
                    s_data[base + j].x = u.x + t_val.x;
                    s_data[base + j].y = u.y + t_val.y;
                    s_data[base + j + half_m].x = u.x - t_val.x;
                    s_data[base + j + half_m].y = u.y - t_val.y;
                }
                __syncthreads();
            }
            
            // For inverse FFT, scale the result by 1/N.
            if (inverse) {
                double scale = 1.0 / N;
                s_data[tid].x *= scale;
                s_data[tid].y *= scale;
            }
            __syncthreads();
            
            // Write the result back to global memory.
            if (tid < N)
                out[tid] = s_data[tid];
        }
        '''
        # Compile the kernel using CuPy's RawModule with NVCC and fast math.
        self.module = cp.RawModule(code=kernel_code,
                                   options=("-use_fast_math",),
                                   backend="nvcc")
        self.fft_shared = self.module.get_function("fft_shared")
    
    def transform(self, x, inverse=False):
        """
        Compute the FFT (or inverse FFT) using the optimized shared-memory kernel.
        
        Args:
            x: Input CuPy array of type cp.complex128 with length N (must be a power of 2).
            inverse: If True, compute the inverse FFT.
        
        Returns:
            A CuPy array of type cp.complex128 containing the FFT result.
        """
        # Ensure input is on the GPU.
        if not isinstance(x, cp.ndarray):
            x = cp.asarray(x)
        N = int(x.size)
        result = cp.empty_like(x)
        # Total shared memory: FFT data (N elements) + twiddle factors (N/2 elements)
        shared_mem_size = x.nbytes + (N // 2) * cp.dtype(cp.complex128).itemsize
        
        self.fft_shared(
            (1,),
            (N,),
            (x.data.ptr, result.data.ptr, cp.int32(N), cp.int32(1 if inverse else 0)),
            shared_mem=shared_mem_size
        )
        return result

if __name__ == "__main__":
    # Instantiate the optimized FFT class.
    fft_cupy = HPDFT_CuPy_Shared_Double_Optimized()
    
    # Set transform size (N must be a power of 2 that fits in one CUDA block).
    N = 1024
    # Generate a random input array of type cp.complex128 directly on the GPU.
    x = cp.random.random(N, dtype=cp.float64) + 1j * cp.random.random(N, dtype=cp.float64)
    
    # Time the forward FFT using CuPy events (data remains on-device).
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    X = fft_cupy.transform(x, inverse=False)
    end.record()
    end.synchronize()
    elapsed_fwd = cp.cuda.get_elapsed_time(start, end)
    print(f"Optimized FFT forward (N={N}) took {elapsed_fwd:.3f} ms.")
    
    # Time the inverse FFT.
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    x_rec = fft_cupy.transform(X, inverse=True)
    end.record()
    end.synchronize()
    elapsed_inv = cp.cuda.get_elapsed_time(start, end)
    print(f"Optimized FFT inverse (N={N}) took {elapsed_inv:.3f} ms.")
    
    # Check the maximum reconstruction error.
    error = cp.max(cp.abs(x - x_rec))
    print(f"Max reconstruction error: {error:.3e}")

import numpy as np
import random
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# CUDA kernel code: Multiply two 256-bit integers.
# Each 256-bit integer is represented as 4 unsigned long long values.
# The product is stored as 8 unsigned long long values.
kernel_code = r"""
// Device function to multiply two 256-bit integers (each represented as 4 64-bit words)
// and store the 512-bit result in an array of 8 64-bit words.
// The multiplication is performed using schoolbook multiplication with proper carry propagation.
__device__ void multiply_256(const unsigned long long *A, const unsigned long long *B, unsigned long long *C)
{
    // Initialize an 8-word (512-bit) product array to zero.
    unsigned long long prod[8] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
    
    // Loop over each word of A and B.
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int k = i + j;
            unsigned long long a = A[i];
            unsigned long long b = B[j];
            // Compute the 128-bit product of a and b:
            // lo holds the low 64 bits; hi is computed using __umul64hi.
            unsigned long long lo = a * b;
            unsigned long long hi = __umul64hi(a, b);
            
            // Add the low part to prod[k] with carry.
            unsigned long long sum = prod[k] + lo;
            unsigned long long carry = (sum < prod[k]) ? 1ULL : 0ULL;
            prod[k] = sum;
            
            // Now add hi and the carry to the next word, propagating the carry.
            int pos = k + 1;
            unsigned long long temp = prod[pos] + hi + carry;
            carry = (temp < prod[pos]) ? 1ULL : 0ULL;
            prod[pos] = temp;
            pos++;
            while (carry != 0 && pos < 8) {
                temp = prod[pos] + carry;
                carry = (temp < prod[pos]) ? 1ULL : 0ULL;
                prod[pos] = temp;
                pos++;
            }
        }
    }
    
    // Write the computed 512-bit product to the output array.
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        C[i] = prod[i];
    }
}

// Kernel: each thread multiplies one pair of 256-bit integers.
// The input arrays d_A and d_B store the 256-bit integers consecutively (4 words each),
// and the output array d_C will store the 512-bit result (8 words each).
__global__ void kernel_multiply_256(const unsigned long long *d_A,
                                    const unsigned long long *d_B,
                                    unsigned long long *d_C,
                                    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        // Get pointers to the idx-th 256-bit numbers in d_A and d_B.
        const unsigned long long *A = d_A + idx * 4;
        const unsigned long long *B = d_B + idx * 4;
        // Get pointer to the output 512-bit result.
        unsigned long long *C = d_C + idx * 8;
        multiply_256(A, B, C);
    }
}
"""

# Compile the kernel
mod = SourceModule(kernel_code)
kernel_multiply_256 = mod.get_function("kernel_multiply_256")

def random_256bit_numbers(num_elements):
    """
    Create a numpy array of shape (num_elements, 4) containing random 256-bit integers.
    Each 256-bit integer is split into 4 unsigned 64-bit words (little-endian order).
    """
    A = np.empty((num_elements, 4), dtype=np.uint64)
    for i in range(num_elements):
        # Use Python's built-in function to generate a random 256-bit integer.
        rand_int = random.getrandbits(256)
        for j in range(4):
            A[i, j] = (rand_int >> (64 * j)) & ((1 << 64) - 1)
    return A

def verify_multiplication(A_host, B_host, C_host):
    """
    Verify that for each index, the 512-bit product computed on the GPU in C_host
    matches the product of the corresponding 256-bit integers computed in Python.
    """
    num_elements = A_host.shape[0]
    for i in range(num_elements):
        # Reconstruct the 256-bit integers from 4 words (little-endian order).
        a_val = 0
        b_val = 0
        for j in range(4):
            a_val += int(A_host[i, j]) << (64 * j)
            b_val += int(B_host[i, j]) << (64 * j)
        product_cpu = a_val * b_val
        
        # Reconstruct the 512-bit product from 8 words.
        product_gpu = 0
        for j in range(8):
            product_gpu += int(C_host[i, j]) << (64 * j)
        
        if product_cpu != product_gpu:
            print(f"Mismatch at index {i}:")
            print(f"CPU product = {product_cpu}")
            print(f"GPU product = {product_gpu}")
            return False
    return True

def main():
    num_elements = 10  # Number of 256-bit integer pairs to multiply.
    # Create random 256-bit numbers for A and B (each with shape (num_elements, 4)).
    A_host = random_256bit_numbers(num_elements)
    B_host = random_256bit_numbers(num_elements)
    
    # Allocate output array for C (shape (num_elements, 8)).
    C_host = np.empty((num_elements, 8), dtype=np.uint64)
    
    # Flatten the arrays to 1D since the kernel expects contiguous memory.
    A_flat = np.ravel(A_host).astype(np.uint64)
    B_flat = np.ravel(B_host).astype(np.uint64)
    C_flat = np.empty(num_elements * 8, dtype=np.uint64)
    
    # Transfer host data to device.
    d_A = drv.mem_alloc(A_flat.nbytes)
    drv.memcpy_htod(d_A, A_flat)
    d_B = drv.mem_alloc(B_flat.nbytes)
    drv.memcpy_htod(d_B, B_flat)
    d_C = drv.mem_alloc(C_flat.nbytes)
    
    # Launch the kernel.
    threads_per_block = 256
    blocks = (num_elements + threads_per_block - 1) // threads_per_block
    kernel_multiply_256(d_A, d_B, d_C, np.int32(num_elements),
                          block=(threads_per_block, 1, 1), grid=(blocks, 1, 1))
    
    # Copy the result back to host.
    drv.memcpy_dtoh(C_flat, d_C)
    C_host = C_flat.reshape((num_elements, 8))
    
    # Verify the multiplication.
    if verify_multiplication(A_host, B_host, C_host):
        print("Multiplication verified successfully for all elements!")
    else:
        print("Multiplication verification failed.")

if __name__ == '__main__':
    main()

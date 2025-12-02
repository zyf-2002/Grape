#include "ioutils.cuh"

void savebin(const string& filename, const void* gpudata, uint size)
{
    // Copy data from GPU to CPU
    void* data = malloc(size);
    cudaMemcpy(data, gpudata, size, cudaMemcpyDeviceToHost);

    // Write data to file
    FILE* file = fopen(filename.c_str(), "wb");
    fwrite(data, 1, size, file);
    fclose(file);
    
    // Free memory
    free(data);

}

ulong findsize(const string& filename)
{
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        fprintf(stderr, "Error: cannot open file %s\n", filename.c_str());
        return 0;
    }

    if (fseeko(file, 0, SEEK_END) != 0) {
        fprintf(stderr, "Error: fseeko failed\n");
        fclose(file);
        return 0;
    }

    off_t size = ftello(file);
    fclose(file);

    if (size < 0) {
        fprintf(stderr, "Error: ftello failed\n");
        return 0;
    }

    return size;
}

void loadbin(const string& filename, void* gpudata, ulong size)
{

    
    // Allocate memory
    void* data = malloc(size);

    // Read data from file
    FILE* file = fopen(filename.c_str(), "rb");

    if (!file) {
        perror(("Error opening file " + filename).c_str());
        exit(1);
    }

    fread(data, 1, size, file);
    fclose(file);
    

    // Copy data from CPU to GPU
    cudaMemcpy(gpudata, data, size, cudaMemcpyHostToDevice);

    // Free memory
    free(data);

}


uint Log2(uint num) {
    if (num == 0) return 0;
    
    // Decrease num to handle the case where num is already a power of 2
    num--;

    uint result = 0;
    
    // Keep shifting the number to the right until it becomes zero. 
    // Each shift means the number is halved, which corresponds to 
    // a division by 2 in logarithmic terms.
    while (num > 0) {
        num >>= 1;
        result++;
    }

    return result;
}
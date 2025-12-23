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

    //void* data = malloc(size);
    void* data;
    cudaMallocHost(&data, size);

    FILE* file = fopen(filename.c_str(), "rb");

    if (!file) {
        perror(("Error opening file " + filename).c_str());
        exit(1);
    }

    fread(data, 1, size, file);
    fclose(file);
    
    // Copy data from CPU to GPU
    CUDA_TIMER_START(cpy);
    cudaMemcpy(gpudata, data, size, cudaMemcpyHostToDevice);
    CUDA_TIMER_STOP(cpy);

    // Free memory
    cudaFreeHost(data);

}

uint load_data(const string& filename, int **out_ptr){
    auto size = findsize(filename) / sizeof(int);
    cout << filename << "size: " << size << endl;   
    //size = size / 2;
    cudaMalloc((void **)out_ptr, sizeof(int) * size);
    loadbin(filename, *out_ptr, sizeof(int) * size);
    return size;
}

void generate_data(int **out_ptr, uint size) {
    // 1. CPU buffer
    std::vector<int> h_tensor(size);

    // 2. 随机数引擎
    std::mt19937 rng(123456);
    std::uniform_int_distribution<int> dist(
        std::numeric_limits<int>::min(),
        std::numeric_limits<int>::max()
    );

    // 3. 生成均匀分布
    for (uint i = 0; i < size; i++) {
        h_tensor[i] = dist(rng);
    }

    // 4. 分配 GPU 内存（注意：是 *out_ptr）
    cudaMalloc((void**)out_ptr, sizeof(int) * size);

    // 5. 拷贝到 GPU（注意：是 *out_ptr）
    cudaMemcpy(*out_ptr, h_tensor.data(),
               sizeof(int) * size,
               cudaMemcpyHostToDevice);
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
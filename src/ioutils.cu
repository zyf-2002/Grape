#include "ioutils.cuh"

void savebin(const string& filename, const void* data, size_t size, bool is_gpu_data)
{
    if (!data || size == 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }

    void* host_data = nullptr;
    
    if (is_gpu_data) {
        // GPU数据：先拷贝到CPU
        host_data = malloc(size);
        if (!host_data) {
            fprintf(stderr, "Error: malloc failed\n");
            return;
        }
        cudaMemcpy(host_data, data, size, cudaMemcpyDeviceToHost);
    } else {
        // CPU数据：直接使用
        host_data = const_cast<void*>(data);
    }
    
    // 打开文件并写入
    FILE* file = fopen(filename.c_str(), "wb");
    if (!file) {
        fprintf(stderr, "Error: Failed to open file %s\n", filename.c_str());
        if (is_gpu_data) free(host_data);
        return;
    }
    
    fwrite(host_data, 1, size, file);
    fclose(file);
    
    // 只有GPU数据需要释放
    if (is_gpu_data) free(host_data);
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

void loadbin(const string& filename, void* data, size_t size, bool is_gpu_data)
{
    if (!data || size == 0) {
        fprintf(stderr, "Error: Invalid input parameters\n");
        return;
    }

    // 读取文件到CPU内存
    void* host_data = malloc(size);
    if (!host_data) {
        fprintf(stderr, "Error: malloc failed\n");
        return;
    }

    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        perror(("Error opening file " + filename).c_str());
        free(host_data);
        return;
    }

    size_t read_count = fread(host_data, 1, size, file);
    fclose(file);

    if (read_count != size) {
        fprintf(stderr, "Error: Read %zu bytes, expected %lu\n", read_count, size);
        free(host_data);
        return;
    }

    if (is_gpu_data) {
        // GPU数据：拷贝到GPU
        cudaMemcpy(data, host_data, size, cudaMemcpyHostToDevice);
    } else {
        // CPU数据：直接拷贝到CPU内存
        memcpy(data, host_data, size);
    }

    free(host_data);
}

uint load_data(const string& filename, int **gpu_ptr){
    auto size = findsize(filename) / sizeof(int);
    //size = size / 2;
    if (*gpu_ptr != nullptr) {
        cudaFree(*gpu_ptr);
        *gpu_ptr = nullptr;
    }
    cudaMalloc((void **)gpu_ptr, sizeof(int) * size);
    int *cpu_ptr = (int*)malloc(sizeof(int) * size);
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        perror(("Error opening file " + filename).c_str());
        exit(1);
    }
    fread(cpu_ptr, 1, size * sizeof(int), file);
    fclose(file);

    cudaMemcpy(*gpu_ptr, cpu_ptr, size * sizeof(int), cudaMemcpyHostToDevice);
    free (cpu_ptr);
    return size;
}

uint load_data(const string& filename, int **gpu_ptr, int **cpu_ptr){
    auto size = findsize(filename) / sizeof(int);
    cout << filename << "size: " << size << endl;   
    //size = size / 2;
    *cpu_ptr = (int*)malloc(sizeof(int) * size);
    cudaMalloc((void **)gpu_ptr, sizeof(int) * size);

    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        perror(("Error opening file " + filename).c_str());
        exit(1);
    }

    fread(*cpu_ptr, sizeof(int), size, file);
    fclose(file);
    
    cudaMemcpy(*gpu_ptr, *cpu_ptr, sizeof(int) * size, cudaMemcpyHostToDevice);
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





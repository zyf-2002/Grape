#include "timer.cuh"

void Timer::start() { start_time = std::chrono::high_resolution_clock::now(); }

double Timer::stop(const std::string& label) 
{
    double duration_sec = elapsed();
    double duration_ms = duration_sec * 1000;  // 转换为毫秒
    printf("\033[35m[Timing]\033[0m %s in: %.3f ms\n", label.c_str(), duration_ms);
    return duration_sec;  // 仍然返回秒（保持接口兼容）
}

double Timer::elapsed() 
{
    auto current_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time);
    return duration.count() / (1e6);
}
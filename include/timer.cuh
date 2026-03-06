#pragma once
#include <iostream>
#include <chrono>

class Timer {
public:
    void start();
    double stop(const std::string& label = "Code block executed");
    double elapsed();

private:
    std::chrono::high_resolution_clock::time_point start_time;
};
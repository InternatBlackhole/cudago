#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <functional>
#include <map>
#include <stdint.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <memory>
#include <optional>


#define log(msg) std::cerr << msg << std::endl
#define logf(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__)

//void report(const char *func, int64_t time, int runNum);
void report(const char *func, int64_t time);

#endif // __TYPES_HPP__
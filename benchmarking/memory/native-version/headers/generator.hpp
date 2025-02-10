#ifndef __GENERATOR_H__
#define __GENERATOR_H__

#include "common.hpp"

typedef struct
{
    const char *name;
    int64_t times;
    std::function<void()> f;
} generatorParam;

typedef struct
{
    std::map<const char *, int> leftToRun;
    int64_t allToRun;

    int nextTest;
    int numTests;
} generatorState;

void logState(generatorState &state);
void logParam(generatorParam &p);
std::function<std::optional<std::function<void()>>()> generator(std::vector<generatorParam>& params, generatorState& state);

#endif // __GENERATOR_H__
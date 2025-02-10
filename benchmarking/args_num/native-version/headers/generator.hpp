#ifndef __GENERATOR_H__
#define __GENERATOR_H__

#include "common.hpp"

void logState(generatorState &state);
void logParam(generatorParam &p);
std::function<std::optional<std::function<void()>>()> generator(std::vector<generatorParam>& params, generatorState& state);

#endif // __GENERATOR_H__
#include "generator.hpp"

void logState(generatorState &state)
{
    logf("state.nextTest: %d\nstate.allToRun: %d\nstate.numTests: %d\n", state.nextTest, state.allToRun, state.numTests);
    for (auto &p : state.leftToRun) {
        logf("name: %s; times: %d\n", p.first, p.second);
    }
}

void logParam(generatorParam &p)
{
    logf("logging param...  name: %s; times: %d\n", p.name, p.times);
}

std::function<std::optional<std::function<void()>>()> generator(std::vector<generatorParam>& params, generatorState& state)
{
    state.allToRun = 0;
    for (int i = 0; i < params.size(); i++)
    {
        generatorParam &p = params[i];
        //logParam(p);
        state.leftToRun[p.name] = p.times;
        state.allToRun += p.times;
    }
    state.nextTest = 0;
    state.numTests = params.size();

    log("Generator state after init in generator:");
    logState(state);

    return [&state, &params]() mutable -> std::optional<std::function<void()>>
    {
        int looped = 0;
        while (true)
        {
            looped++;
            if (looped > 1000000) {
                log("Looped too many times in generator. exiting...");
                logState(state);
                exit(1);
            }
            if (state.allToRun == 0)
            {
                return std::nullopt;
            }
            generatorParam &p = params[state.nextTest];
            if (state.leftToRun[p.name] > 0)
            {
                state.leftToRun[p.name]--;
                state.allToRun--;
                state.nextTest = (state.nextTest + 1) % state.numTests;
                return p.f;
            }
            state.nextTest = (state.nextTest + 1) % state.numTests;
        }
        return std::nullopt;
    };
}
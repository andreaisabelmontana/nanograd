// A minimal multilayer perceptron built on the autograd Value. Parameters are persistent leaf
// Values; each forward pass builds a fresh graph over them, so backward() fills parameter grads
// and a step of SGD just nudges each parameter's data.
#pragma once

#include "value.hpp"

#include <random>
#include <vector>

namespace nanograd {

class Neuron {
public:
    Neuron(int nin, std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        w_.reserve(nin);
        for (int i = 0; i < nin; ++i) {
            w_.emplace_back(dist(rng));
        }
    }

    // tanh(w . x + b)
    Value operator()(const std::vector<Value>& x) const {
        Value act = b_;
        for (std::size_t i = 0; i < w_.size(); ++i) {
            act = act + w_[i] * x[i];
        }
        return act.tanh();
    }

    void collect(std::vector<Value>& out) {
        for (auto& wi : w_) {
            out.push_back(wi);
        }
        out.push_back(b_);
    }

private:
    std::vector<Value> w_;
    Value b_{0.0};
};

class Layer {
public:
    Layer(int nin, int nout, std::mt19937& rng) {
        neurons_.reserve(nout);
        for (int i = 0; i < nout; ++i) {
            neurons_.emplace_back(nin, rng);
        }
    }

    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> out;
        out.reserve(neurons_.size());
        for (const auto& n : neurons_) {
            out.push_back(n(x));
        }
        return out;
    }

    void collect(std::vector<Value>& out) {
        for (auto& n : neurons_) {
            n.collect(out);
        }
    }

private:
    std::vector<Neuron> neurons_;
};

class MLP {
public:
    MLP(int nin, const std::vector<int>& nouts, std::mt19937& rng) {
        int prev = nin;
        for (int n : nouts) {
            layers_.emplace_back(prev, n, rng);
            prev = n;
        }
    }

    std::vector<Value> operator()(std::vector<Value> x) const {
        for (const auto& l : layers_) {
            x = l(x);
        }
        return x;
    }

    std::vector<Value> parameters() {
        std::vector<Value> out;
        for (auto& l : layers_) {
            l.collect(out);
        }
        return out;
    }

private:
    std::vector<Layer> layers_;
};

} // namespace nanograd

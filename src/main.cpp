// Trains a small MLP to learn XOR using nothing but the autograd engine, printing the loss
// curve and the learned outputs. Run it: the loss should fall toward zero and the four
// predictions should match their targets' signs.
#include "nanograd/mlp.hpp"

#include <cstdio>
#include <random>
#include <vector>

using nanograd::MLP;
using nanograd::Value;

int main() {
    std::mt19937 rng(1337);
    MLP net(2, {8, 8, 1}, rng); // 2 inputs -> 8 -> 8 -> 1, all tanh

    const std::vector<std::vector<double>> xs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<double> ys = {-1, 1, 1, -1}; // XOR, encoded in {-1, +1}

    auto params = net.parameters();
    const double lr = 0.05;
    const int epochs = 500;

    std::printf("training XOR on a %zu-parameter MLP\n\n", params.size());
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        Value loss(0.0); // sum of squared error over the batch
        for (std::size_t i = 0; i < xs.size(); ++i) {
            std::vector<Value> x = {Value(xs[i][0]), Value(xs[i][1])};
            Value diff = net(x)[0] - Value(ys[i]);
            loss = loss + diff * diff;
        }

        for (auto& p : params) {
            p.zero_grad();
        }
        loss.backward();
        for (auto& p : params) {
            p.set_data(p.data() - lr * p.grad()); // SGD step
        }

        if (epoch % 50 == 0) {
            std::printf("epoch %3d   loss %.6f\n", epoch, loss.data());
        }
    }

    std::printf("\nlearned XOR:\n");
    for (std::size_t i = 0; i < xs.size(); ++i) {
        std::vector<Value> x = {Value(xs[i][0]), Value(xs[i][1])};
        double pred = net(x)[0].data();
        std::printf("  [%g, %g] -> % .3f   (target % g)\n", xs[i][0], xs[i][1], pred, ys[i]);
    }
    return 0;
}

// Tests for the autograd engine: forward composition, gradients checked against finite
// differences, and an end-to-end XOR training run. No framework — a tiny assert harness that
// returns non-zero on failure so ctest/CI can gate on it.
#include "nanograd/mlp.hpp"
#include "nanograd/value.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using nanograd::MLP;
using nanograd::Value;

static int failures = 0;

#define CHECK(cond, msg)                          \
    do {                                          \
        if (!(cond)) {                            \
            std::printf("FAIL: %s\n", (msg));     \
            ++failures;                           \
        }                                         \
    } while (0)

#define CHECK_NEAR(a, b, tol, msg)                                                    \
    do {                                                                              \
        double da = (a), db = (b);                                                    \
        if (std::fabs(da - db) > (tol)) {                                             \
            std::printf("FAIL: %s (%.8f vs %.8f)\n", (msg), da, db);                  \
            ++failures;                                                               \
        }                                                                             \
    } while (0)

static void test_forward() {
    Value a(2.0), b(-3.0), c(10.0);
    Value f = (a * b + c) * Value(2.0); // ((2*-3)+10)*2 = 8
    CHECK_NEAR(f.data(), 8.0, 1e-9, "forward composition");
}

static void test_gradients_match_finite_differences() {
    // y = tanh(a*b + a); a is used twice, exercising gradient accumulation.
    auto y = [](double a, double b) { return std::tanh(a * b + a); };
    const double A = 0.7, B = -1.3, h = 1e-6;

    Value a(A), b(B);
    Value out = (a * b + a).tanh();
    a.zero_grad();
    b.zero_grad();
    out.backward();

    double ga = (y(A + h, B) - y(A - h, B)) / (2 * h);
    double gb = (y(A, B + h) - y(A, B - h)) / (2 * h);
    CHECK_NEAR(a.grad(), ga, 1e-4, "d/da matches finite difference");
    CHECK_NEAR(b.grad(), gb, 1e-4, "d/db matches finite difference");
}

static void test_xor_converges() {
    std::mt19937 rng(1337);
    MLP net(2, {8, 8, 1}, rng);
    const std::vector<std::vector<double>> xs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<double> ys = {-1, 1, 1, -1};
    auto params = net.parameters();

    double loss = 0.0;
    for (int epoch = 0; epoch < 500; ++epoch) {
        Value l(0.0);
        for (std::size_t i = 0; i < xs.size(); ++i) {
            std::vector<Value> x = {Value(xs[i][0]), Value(xs[i][1])};
            Value diff = net(x)[0] - Value(ys[i]);
            l = l + diff * diff;
        }
        for (auto& p : params) {
            p.zero_grad();
        }
        l.backward();
        for (auto& p : params) {
            p.set_data(p.data() - 0.05 * p.grad());
        }
        loss = l.data();
    }

    CHECK(loss < 0.1, "XOR training drives loss below 0.1");
    int correct = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        std::vector<Value> x = {Value(xs[i][0]), Value(xs[i][1])};
        double pred = net(x)[0].data();
        if ((pred > 0) == (ys[i] > 0)) {
            ++correct;
        }
    }
    CHECK(correct == 4, "all four XOR predictions have the correct sign");
}

int main() {
    test_forward();
    test_gradients_match_finite_differences();
    test_xor_converges();
    if (failures == 0) {
        std::printf("all tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}

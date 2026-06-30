// nanograd — a tiny reverse-mode automatic-differentiation engine over scalars.
//
// A Value wraps a node in a dynamically-built computation graph. Arithmetic and activation
// operations create new nodes that remember how to push gradients back to their inputs; calling
// backward() on a node walks the graph in reverse-topological order and fills in every grad.
// This is the same idea that powers PyTorch/autograd, reduced to its essence.
#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <unordered_set>
#include <vector>

namespace nanograd {

struct Node {
    double data = 0.0;
    double grad = 0.0;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward = [] {}; // local rule: push this node's grad to its parents
};

class Value {
public:
    Value(double d = 0.0) : n_(std::make_shared<Node>()) { n_->data = d; } // NOLINT: implicit by design
    explicit Value(std::shared_ptr<Node> n) : n_(std::move(n)) {}

    double data() const { return n_->data; }
    double grad() const { return n_->grad; }
    void set_data(double d) { n_->data = d; }
    void zero_grad() { n_->grad = 0.0; }
    const std::shared_ptr<Node>& node() const { return n_; }

    Value operator+(const Value& o) const {
        auto out = make({n_, o.n_}, n_->data + o.n_->data);
        auto a = n_, b = o.n_, c = out;
        c->backward = [a, b, c] {
            a->grad += c->grad;
            b->grad += c->grad;
        };
        return Value(out);
    }

    Value operator*(const Value& o) const {
        auto out = make({n_, o.n_}, n_->data * o.n_->data);
        auto a = n_, b = o.n_, c = out;
        c->backward = [a, b, c] {
            a->grad += b->data * c->grad;
            b->grad += a->data * c->grad;
        };
        return Value(out);
    }

    Value operator-() const { return *this * Value(-1.0); }
    Value operator-(const Value& o) const { return *this + (-o); }

    Value pow(double e) const {
        auto out = make({n_}, std::pow(n_->data, e));
        auto a = n_, c = out;
        c->backward = [a, c, e] { a->grad += e * std::pow(a->data, e - 1.0) * c->grad; };
        return Value(out);
    }

    Value operator/(const Value& o) const { return *this * o.pow(-1.0); }

    Value tanh() const {
        double t = std::tanh(n_->data);
        auto out = make({n_}, t);
        auto a = n_, c = out;
        c->backward = [a, c, t] { a->grad += (1.0 - t * t) * c->grad; };
        return Value(out);
    }

    Value relu() const {
        auto out = make({n_}, n_->data > 0.0 ? n_->data : 0.0);
        auto a = n_, c = out;
        c->backward = [a, c] { a->grad += (c->data > 0.0 ? 1.0 : 0.0) * c->grad; };
        return Value(out);
    }

    // Reverse-mode autodiff: seed this node's gradient with 1 and propagate to every ancestor.
    void backward() {
        std::vector<std::shared_ptr<Node>> topo;
        std::unordered_set<Node*> seen;
        build_topo(n_, seen, topo);
        n_->grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward();
        }
    }

private:
    static std::shared_ptr<Node> make(std::vector<std::shared_ptr<Node>> parents, double data) {
        auto n = std::make_shared<Node>();
        n->data = data;
        n->parents = std::move(parents);
        return n;
    }

    static void build_topo(const std::shared_ptr<Node>& v, std::unordered_set<Node*>& seen,
                           std::vector<std::shared_ptr<Node>>& topo) {
        if (!seen.insert(v.get()).second) {
            return;
        }
        for (const auto& p : v->parents) {
            build_topo(p, seen, topo);
        }
        topo.push_back(v);
    }

    std::shared_ptr<Node> n_;
};

} // namespace nanograd

// neuron_structure.cpp
// Single-file in-memory Neuron data structure (no DB).
// C++17 required.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using String = std::string;

// ---------- utility: generate a simple UUID-like string ----------
static String generate_uuid_v4() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dist;
    uint64_t a = dist(gen);
    uint64_t b = dist(gen);
    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(8) << ((uint32_t)(a >> 32)) << "-"
        << std::setw(4) << ((uint16_t)(a >> 16)) << "-"
        << std::setw(4) << ((uint16_t)a) << "-"
        << std::setw(4) << ((uint16_t)(b >> 48)) << "-"
        << std::setw(12) << (b & 0xFFFFFFFFFFFFULL);
    return oss.str();
}

// ---------- escape function for serialization ----------
static String escape(const String &s) {
    String out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == '=' || c == ';' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

// ---------- Neuron class ----------
struct Neuron {
    String id;
    std::map<String, String> data;
    std::map<String, String> metadata;
    std::set<String> links;

    std::function<bool(const Neuron&, const Neuron&)> link_rule = nullptr;

    Neuron() : id(generate_uuid_v4()) {}
    explicit Neuron(const String &custom_id) : id(custom_id) {}

    void set_data(const String &k, const String &v) { data[k] = v; }
    bool has_data(const String &k) const { return data.find(k) != data.end(); }
    String get_data(const String &k, const String &def = "") const {
        auto it = data.find(k);
        return it == data.end() ? def : it->second;
    }

    void set_meta(const String &k, const String &v) { metadata[k] = v; }
    String get_meta(const String &k, const String &def = "") const {
        auto it = metadata.find(k);
        return it == metadata.end() ? def : it->second;
    }

    String brief() const {
        std::ostringstream oss;
        oss << "Neuron(" << id << ") {";
        bool first = true;
        for (auto &p : data) {
            if (!first) oss << ", ";
            oss << p.first << ":" << p.second;
            first = false;
        }
        oss << "}";
        return oss.str();
    }
};

// ---------- Manager: NeuronNetwork ----------
class NeuronNetwork {
private:
    std::unordered_map<String, Neuron> neurons;
    mutable std::mutex mtx;

public:
    NeuronNetwork() = default;

    String create_neuron() {
        std::lock_guard<std::mutex> lk(mtx);
        Neuron n;
        String id = n.id;
        neurons.emplace(id, std::move(n));
        return id;
    }

    String create_neuron(const std::map<String, String>& data, const std::map<String, String>& metadata) {
        std::lock_guard<std::mutex> lk(mtx);
        Neuron n;
        n.data = data;
        n.metadata = metadata;
        String id = n.id;
        neurons.emplace(id, std::move(n));
        return id;
    }

    bool remove_neuron(const String &id) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = neurons.find(id);
        if (it == neurons.end()) return false;
        for (const auto &nbr_id : it->second.links) {
            auto nit = neurons.find(nbr_id);
            if (nit != neurons.end()) nit->second.links.erase(id);
        }
        neurons.erase(it);
        return true;
    }

    bool get_neuron_copy(const String &id, Neuron &out) const {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = neurons.find(id);
        if (it == neurons.end()) return false;
        out = it->second;
        return true;
    }

    bool set_link_rule(const String &id, std::function<bool(const Neuron&, const Neuron&)> rule) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = neurons.find(id);
        if (it == neurons.end()) return false;
        it->second.link_rule = std::move(rule);
        return true;
    }

    bool link_allowed_locked(const Neuron &a, const Neuron &b) const {
        if (a.link_rule && !a.link_rule(a,b)) return false;
        if (b.link_rule && !b.link_rule(b,a)) return false;
        return true;
    }

    bool link(const String &a_id, const String &b_id) {
        if (a_id == b_id) return false;
        std::lock_guard<std::mutex> lk(mtx);
        auto ita = neurons.find(a_id);
        auto itb = neurons.find(b_id);
        if (ita == neurons.end() || itb == neurons.end()) return false;
        if (!link_allowed_locked(ita->second, itb->second)) return false;
        ita->second.links.insert(b_id);
        itb->second.links.insert(a_id);
        return true;
    }

    bool unlink(const String &a_id, const String &b_id) {
        std::lock_guard<std::mutex> lk(mtx);
        auto ita = neurons.find(a_id);
        auto itb = neurons.find(b_id);
        if (ita == neurons.end() || itb == neurons.end()) return false;
        ita->second.links.erase(b_id);
        itb->second.links.erase(a_id);
        return true;
    }

    std::vector<String> neighbors(const String &id) const {
        std::lock_guard<std::mutex> lk(mtx);
        std::vector<String> out;
        auto it = neurons.find(id);
        if (it == neurons.end()) return out;
        out.insert(out.end(), it->second.links.begin(), it->second.links.end());
        return out;
    }

    std::set<String> traverse_dfs(const String &start_id, size_t max_visit = 100000) const {
        std::lock_guard<std::mutex> lk(mtx);
        std::set<String> visited;
        if (neurons.find(start_id) == neurons.end()) return visited;
        std::vector<String> stack { start_id };
        while (!stack.empty() && visited.size() < max_visit) {
            String cur = stack.back(); stack.pop_back();
            if (visited.count(cur)) continue;
            visited.insert(cur);
            auto it = neurons.find(cur);
            if (it == neurons.end()) continue;
            for (auto &nbr : it->second.links) if (!visited.count(nbr)) stack.push_back(nbr);
        }
        return visited;
    }

    std::unordered_map<String,int> traverse_bfs_distance(const String &start_id, int max_depth=1000) const {
        std::lock_guard<std::mutex> lk(mtx);
        std::unordered_map<String,int> dist;
        if (neurons.find(start_id) == neurons.end()) return dist;
        std::vector<String> q { start_id };
        dist[start_id] = 0;
        size_t idx = 0;
        while (idx < q.size()) {
            String cur = q[idx++];
            int d = dist[cur];
            if (d >= max_depth) continue;
            auto it = neurons.find(cur);
            if (it == neurons.end()) continue;
            for (auto &nbr : it->second.links) {
                if (!dist.count(nbr)) { dist[nbr] = d+1; q.push_back(nbr); }
            }
        }
        return dist;
    }

    String serialize_all() const {
        std::lock_guard<std::mutex> lk(mtx);
        std::ostringstream oss;
        for (const auto &p : neurons) {
            const Neuron &n = p.second;
            oss << "ID:" << n.id << "\nDATA:";
            bool first = true;
            for (auto &kv : n.data) {
                if (!first) oss << ";";
                oss << escape(kv.first) << "=" << escape(kv.second);
                first = false;
            }
            oss << "\nMETA:";
            first = true;
            for (auto &kv : n.metadata) {
                if (!first) oss << ";";
                oss << escape(kv.first) << "=" << escape(kv.second);
                first = false;
            }
            oss << "\nLINKS:";
            first = true;
            for (auto &lid : n.links) { if (!first) oss << ","; oss << lid; first=false; }
            oss << "\n---\n";
        }
        return oss.str();
    }

    std::vector<String> list_ids() const {
        std::lock_guard<std::mutex> lk(mtx);
        std::vector<String> out;
        out.reserve(neurons.size());
        for (auto &p : neurons) out.push_back(p.first);
        return out;
    }
};

// ---------------- Demo ----------------
int main() {
    NeuronNetwork net;

    String nroot = net.create_neuron();
    String nmedia = net.create_neuron();
    String nphoto = net.create_neuron();
    String nvideo = net.create_neuron();

    // Set link rule
    net.set_link_rule(nmedia, [](const Neuron &self, const Neuron &other) {
        auto it = other.metadata.find("layer");
        if (it == other.metadata.end()) return true;
        return it->second == "filesystem";
    });

    // Link neurons
    net.link(nroot, nmedia);
    net.link(nmedia, nphoto);
    net.link(nmedia, nvideo);

    // Traverse DFS
    auto visited = net.traverse_dfs(nmedia);
    std::cout << "DFS visited " << visited.size() << " nodes\n";

    // BFS distances
    auto dist = net.traverse_bfs_distance(nroot);
    std::cout << "BFS distances from root:\n";
    for (auto &p : dist) std::cout << " - " << p.first << " : " << p.second << "\n";

    // Serialize
    std::cout << "--- Serialized network ---\n" << net.serialize_all();

    return 0;
}

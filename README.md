# Neuron Data Structure (C++17)

A brain-inspired in-memory neuron network implemented in C++17.
Each neuron can store data, metadata, and bidirectional links. Supports custom link rules, graph traversals, and network serialization for debugging.

Features
Neurons with unique IDs: Automatically generated UUID-like IDs.
Flexible data storage: Each neuron has data and metadata key-value maps.
Bidirectional links: Connect neurons with validation rules.
Custom link rules: Each neuron can define local rules to allow or reject links.
Graph traversal: Supports DFS and BFS traversal with distance calculation.
Serialization: Convert the entire network to a human-readable string.
Thread-safe: Uses std::mutex for safe multi-threaded access.

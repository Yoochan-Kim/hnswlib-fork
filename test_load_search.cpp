#include "hnswlib/hnswlib.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <set>
#include <iostream>

bool LoadBinaryVectors(const std::string& filename, std::vector<float>& vectors, int& dim, int& num_vectors) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&num_vectors), sizeof(int));
    if (file.fail()) {
        std::cerr << "Failed to read header from " << filename << std::endl;
        return false;
    }

    vectors.resize(static_cast<size_t>(dim) * num_vectors);
    file.read(reinterpret_cast<char*>(vectors.data()), vectors.size() * sizeof(float));
    if (file.fail()) {
        std::cerr << "Failed to read vectors from " << filename << std::endl;
        return false;
    }

    file.close();
    return true;
}

bool LoadGroundTruth(const std::string& filename, std::vector<int64_t>& ground_truth, int& k, int& num_queries) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    file.read(reinterpret_cast<char*>(&k), sizeof(int));
    file.read(reinterpret_cast<char*>(&num_queries), sizeof(int));
    if (file.fail()) {
        std::cerr << "Failed to read header from " << filename << std::endl;
        return false;
    }

    ground_truth.resize(static_cast<size_t>(k) * num_queries);
    file.read(reinterpret_cast<char*>(ground_truth.data()), ground_truth.size() * sizeof(int64_t));
    if (file.fail()) {
        std::cerr << "Failed to read ground truth from " << filename << std::endl;
        return false;
    }

    file.close();
    return true;
}

int main() {
    int d = 1024;
    int nb = 1000000;
    int M = 64;
    int ef_construction = 200;
    int ef_search = 100;
    int nq = 100;
    int k = 10;

    // InnerProductSpace for cosine similarity
    hnswlib::InnerProductSpace space(d);

    // Load index
    std::string index_path = "index0.bin";
    std::cout << "Loading index from '" << index_path << "'...\n";
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, index_path);

    // Set ef for search
    alg_hnsw->setEf(ef_search);

    // Load test queries
    std::vector<float> all_queries;
    int total_queries;
    int query_dims;
    if (!LoadBinaryVectors("bioasq_test_queries.bin", all_queries, query_dims, total_queries)) {
        std::cerr << "Failed to load bioasq_test_queries.bin" << std::endl;
        delete alg_hnsw;
        return 1;
    }

    // Load ground truth
    std::vector<int64_t> parquet_ground_truth;
    int gt_k, gt_queries;
    if (LoadGroundTruth("bioasq_ground_truth.bin", parquet_ground_truth, gt_k, gt_queries)) {
        // Adjust parameters to available data
        k = std::min(k, gt_k);
        nq = std::min(nq, gt_queries);

        // Slice ground truth: take first k neighbors for first nq queries
        std::vector<int64_t> sliced_gt(nq * k);
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < k; ++j) {
                sliced_gt[i * k + j] = parquet_ground_truth[i * gt_k + j];
            }
        }
        parquet_ground_truth = std::move(sliced_gt);
    } else {
        nq = std::min(nq, total_queries);  // Fallback when no GT
    }

    // Slice queries to match final nq
    std::vector<float> queries(all_queries.begin(), all_queries.begin() + nq * d);

    std::cout << "Setup complete: " << nb << "x" << d << " vectors, "
              << nq << " queries, k=" << k << std::endl;

    // Perform search
    std::cout << "Searching with ef=" << ef_search << "...\n\n";

    std::vector<double> query_times(nq);
    std::vector<double> query_recalls(nq);

    for (int i = 0; i < nq; i++) {
        // Measure search time
        auto query_start = std::chrono::high_resolution_clock::now();
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw->searchKnn(queries.data() + i * d, k);
        auto query_end = std::chrono::high_resolution_clock::now();

        query_times[i] = std::chrono::duration<double>(query_end - query_start).count() * 1000.0;

        // Extract search results
        std::set<int64_t> result_set;
        while (!result.empty()) {
            result_set.insert(static_cast<int64_t>(result.top().second));
            result.pop();
        }

        // Calculate recall by comparing with ground truth
        int correct = 0;
        for (int j = 0; j < k; ++j) {
            if (result_set.count(parquet_ground_truth[i * k + j])) {
                correct++;
            }
        }
        query_recalls[i] = static_cast<double>(correct) / k;

        std::cout << "Query " << i << ": " << query_times[i] << " ms, Recall@" << k << ": " << query_recalls[i] << "\n";
    }

    // Calculate statistics
    double total_time = 0.0;
    double total_recall = 0.0;
    double min_time = query_times[0];
    double max_time = query_times[0];

    for (int i = 0; i < nq; i++) {
        total_time += query_times[i];
        total_recall += query_recalls[i];
        if (query_times[i] < min_time) min_time = query_times[i];
        if (query_times[i] > max_time) max_time = query_times[i];
    }

    double avg_time = total_time / nq;
    double avg_recall = total_recall / nq;

    std::cout << "\n========== Statistics ==========\n";
    std::cout << "Total queries: " << nq << "\n";
    std::cout << "k: " << k << "\n";
    std::cout << "ef: " << ef_search << "\n";
    std::cout << "\nTime Statistics:\n";
    std::cout << "  Total time: " << total_time << " ms\n";
    std::cout << "  Average time: " << avg_time << " ms\n";
    std::cout << "  Min time: " << min_time << " ms\n";
    std::cout << "  Max time: " << max_time << " ms\n";
    std::cout << "\nRecall Statistics:\n";
    std::cout << "  Average Recall@" << k << ": " << avg_recall << "\n";
    std::cout << "================================\n";

    delete alg_hnsw;
    return 0;
}

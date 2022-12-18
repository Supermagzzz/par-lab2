#include <iostream>
#include <chrono>
#include <functional>
#include <vector>
#include <random>
#include <queue>
#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <atomic>

using namespace std;

int n, m;
vector<vector<int>> g;
mt19937 rnd(time(nullptr));

void genRandom() {
    n = (int)(rnd() % 100 + 1);
    m = (int)(rnd() % (n * n) + 1);
    g.assign(n, vector<int>(0));
    for (int i = 0; i < m; i++) {
        int a = (int)(rnd() % n);
        int b = (int)(rnd() % n);
        g[a].push_back(b);
    }
}

void genSample() {
    int maxSize = 500;
    auto getNum = [&](int i, int j, int z) {
        return i * maxSize * maxSize + j * maxSize + z;
    };
    auto check = [&](int x) {
        return 0 <= x && x < maxSize;
    };
    n = maxSize * maxSize * maxSize;
    g.assign(n, vector<int>(0));
    m = 0;
    for (int i = 0; i < maxSize; i++) {
        for (int j = 0; j < maxSize; j++) {
            for (int z = 0; z < maxSize; z++) {
                if (check(i + 1)) {
                    g[getNum(i, j, z)].push_back(getNum(i + 1, j, z));
                    m++;
                }
                if (check(j + 1)) {
                    g[getNum(i, j, z)].push_back(getNum(i, j + 1, z));
                    m++;
                }
                if (check(z + 1)) {
                    g[getNum(i, j, z)].push_back(getNum(i, j, z + 1));
                    m++;
                }
            }
        }
    }
}

vector<int> bfs() {
    vector<int> dist(n, -1);
    queue<int> q;
    q.push(0);
    dist[0] = 0;
    while (!q.empty()) {
        int x = q.front();
        q.pop();
        for (int i : g[x]) {
            if (dist[i] == -1) {
                dist[i] = dist[x] + 1;
                q.push(i);
            }
        }
    }
    return dist;
}

struct FrontierElement {
    int x, size;
};

vector<int> tmpScanSize;
vector<int> tmpPosScanAndFilter, positionScanAndFilter;
void init() {
    tmpScanSize = vector<int>();
    tmpPosScanAndFilter = vector<int>();
    positionScanAndFilter = vector<int>();
}

void scanSize(vector<FrontierElement>& arr, int size) {
    if (tmpScanSize.size() < size) {
        tmpScanSize.resize(size);
    }
    for (int j = 1; j < size; j *= 2) {
        #pragma omp parallel for
        for (int i = j; i < size; i++) {
            tmpScanSize[i] = arr[i].size + arr[i - j].size;
        }
        #pragma omp parallel for
        for (int i = j; i < size; i++) {
            arr[i].size = tmpScanSize[i];
        }
    }
}

int scanAndFilter(vector<FrontierElement>& old, int size, vector<FrontierElement>& nw) {
    if (positionScanAndFilter.size() < size) {
        positionScanAndFilter.resize(size);
        tmpPosScanAndFilter.resize(size);
    }
    {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            positionScanAndFilter[i] = old[i].size > 0;
        }
    }
    for (int j = 1; j < size; j *= 2) {
        #pragma omp parallel for
        for (int i = j; i < size; i++) {
            tmpPosScanAndFilter[i] = positionScanAndFilter[i] + positionScanAndFilter[i - j];
        }
        #pragma omp parallel for
        for (int i = j; i < size; i++) {
            positionScanAndFilter[i] = tmpPosScanAndFilter[i];
        }
    }
    if (nw.size() < positionScanAndFilter[size - 1]) {
        nw.resize(positionScanAndFilter[size - 1]);
    }
    {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            int last = i == 0 ? 0 : positionScanAndFilter[i - 1];
            if (positionScanAndFilter[i] != last) {
                nw[positionScanAndFilter[i] - 1] = old[i];
            }
            old[i].size = 0;
        }
    }
    return positionScanAndFilter[size - 1];
}

vector<atomic<int>> parallelBfs() {
    omp_set_num_threads(4);
    omp_set_nested(1);
    init();

    vector<atomic<int>> dist(n);
    {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            dist[i] = -1;
        }
    }
    dist[0] = 0;

    vector<FrontierElement> frontier(1);
    frontier[0] = {0, (int)g[0].size()};
    int frontierSize = 1;

    vector<FrontierElement> newFrontier;

    #pragma omp single
    {
        while (frontierSize > 0) {
            auto [_, newFrontierSize] = frontier[frontierSize - 1];
            if (newFrontier.size() < newFrontierSize) {
                newFrontier.resize(newFrontierSize);
            }
            {
                #pragma omp parallel for
                for (int i = 0; i < frontierSize; i++) {
                    const auto& [x, pos] = frontier[i];
                    {
                        for (int j = 0; j < (int)g[x].size(); j++) {
                            int y = g[x][j];
                            int expected = -1;
                            if (dist[y].compare_exchange_strong(expected, dist[x] + 1)) {
                                newFrontier[pos - j - 1] = {y, (int)g[y].size()};
                            }
                        }
                    }
                }
            }
            frontierSize = scanAndFilter(newFrontier, newFrontierSize, frontier);
            scanSize(frontier, frontierSize);
        }
    }
    return dist;
}

bool eq(const auto& a, const auto& b) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

void stress() {
    while (true) {
        genRandom();
        const auto& realAns = bfs();
        const auto& myAns = parallelBfs();
        if (!eq(realAns, myAns)) {
            cout << "NO" << endl;
            exit(0);
        }
        cout << "YES" << endl;
    }
}

void benchmark() {
    double sum_simple = 0;
    double sum_parallel = 0;
    genSample();
    cerr << "run" << endl;
    for (int i = 0; i < 5; i++) {
        std::chrono::steady_clock::time_point begin_simple = std::chrono::steady_clock::now();
        const auto& realAns = bfs();
        std::chrono::steady_clock::time_point end_simple = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point begin_par = std::chrono::steady_clock::now();
        const auto& myAns = parallelBfs();
        std::chrono::steady_clock::time_point end_par = std::chrono::steady_clock::now();

        auto simple_time = std::chrono::duration_cast<std::chrono::microseconds>(end_simple - begin_simple).count();
        auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end_par - begin_par).count();
        cout << simple_time / 1e6 << " " << parallel_time / 1e6 << " " << simple_time * 1.0 / parallel_time << endl;

        sum_simple += simple_time / 1e6;
        sum_parallel += parallel_time / 1e6;

        assert(eq(realAns, myAns));
    }
    cout << "avg" << endl;
    cout << sum_simple / sum_parallel << endl;
}

int main() {
    benchmark();
    return 0;
}

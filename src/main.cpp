#include <sycl/sycl.hpp>
#include <iostream>
#include "myBuffer.cpp"
#include "myHostAccessor.cpp"
#include "myAccessor.cpp"

using namespace sycl;

int main() {

    queue q;

    constexpr size_t N = 10;

    myBuffer<int> buf(q, N);

    myHostAccessor host_acc(buf);
    for (size_t i = 0; i < N; ++i) {
        host_acc[i] = i;
    }
    buf.copy_host_to_device();

    q.submit([&](handler& h) {
        myAccessor acc(buf);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc[idx] += 1;
        });
    }).wait();

    buf.copy_device_to_host();

    for (size_t i = 0; i < N; ++i) {
        std::cout << host_acc[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

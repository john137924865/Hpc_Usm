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

    {
        myHostAccessor host_acc(buf, access::mode::write);
        for (size_t i = 0; i < N; ++i) {
            host_acc[i] = i;
        }
        buf.copy_host_to_device();
    }

    event e1 = q.submit([&](handler& h) {
        myAccessor acc(buf, h, access::mode::read);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc[idx];
        });
    });
    buf.add_event(e1);

    event e2 = q.submit([&](handler& h) {
        myAccessor acc(buf, h, access::mode::read);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc[idx];
        });
    });
    buf.add_event(e2);

    event e3 = q.submit([&](handler& h) {
        myAccessor acc(buf, h);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc[idx] += 1;
        });
    });
    buf.add_event(e3);

    {
        buf.copy_device_to_host();
        myHostAccessor host_acc(buf, access::mode::read);
        for (size_t i = 0; i < N; ++i) {
            std::cout << host_acc[i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

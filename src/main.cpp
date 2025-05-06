#include <sycl/sycl.hpp>
#include <iostream>
#include "myBuffer.cpp"
#include "myHostAccessor.cpp"
#include "myAccessor.cpp"

using namespace sycl;

int main() {

    queue q;

    constexpr size_t N = 10;
    
    // buffer non inizializzato
    myBuffer<int> buf(q, N);

    // Inizializzazione con host_accessor
    myHostAccessor host_acc(buf);
    for (size_t i = 0; i < N; ++i) {
        host_acc[i] = i;
    }

    // Copia i dati da host a device internamente al buffer
    auto e1 = buf.copy_host_to_device(q);

    // aspetto che termini la copia
    auto e2 = q.submit([&](handler& h) {
        h.depends_on(e1);
        myAccessor acc(buf);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc[idx] += 1;
        });
    });

    // Copia i dati da device a host internamente al buffer al termine del kernel, quindi aspetto
    buf.copy_device_to_host(q, e2).wait();

    for (size_t i = 0; i < N; ++i) {
        std::cout << host_acc[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

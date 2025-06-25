#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "myBuffer.cpp"
#include "myHostAccessor.cpp"
#include "myAccessor.cpp"

void test1();
void test2();

constexpr size_t N = 10000000;

int main() {
    test1();
    test2();
}

void test1() {

    auto start = std::chrono::high_resolution_clock::now();

    sycl::queue q;

    myBuffer<int> a(q, N, "a");
    myBuffer<int> b(q, N, "b");
    myBuffer<int> c(q, N, "c");
    myBuffer<int> d(q, N, "d");
    myBuffer<int> e(q, N, "e");
    myBuffer<int> f(q, N, "f");
    myBuffer<int> g(q, N, "g");
    myBuffer<int> h(q, N, "h");

    {
        myHostAccessor host_acc_a(a, sycl::access::mode::write);
        myHostAccessor host_acc_b(b, sycl::access::mode::write);
        myHostAccessor host_acc_d(d, sycl::access::mode::write);
        myHostAccessor host_acc_f(f, sycl::access::mode::write);
        for (size_t i = 0; i < N; ++i) {
            host_acc_a[i] = i;
            host_acc_b[i] = i;
            host_acc_d[i] = i;
            host_acc_f[i] = i;
        }
        a.copy_host_to_device();
        b.copy_host_to_device();
        d.copy_host_to_device();
        f.copy_host_to_device();
    }

    std::cout << std::endl << " - a_b_c" << std::endl;

    sycl::event a_b_c = q.submit([&](sycl::handler& h) {
        myAccessor acc_a(a, h, sycl::access::mode::read);
        myAccessor acc_b(b, h, sycl::access::mode::read);
        myAccessor acc_c(c, h, sycl::access::mode::write);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            acc_c[idx] = acc_a[idx] + acc_b[idx];
        });
    });
    a.add_event(a_b_c, "a_b_c");
    b.add_event(a_b_c, "a_b_c");
    c.add_event(a_b_c, "a_b_c");

    std::cout << std::endl << " - c_d_e" << std::endl;

    sycl::event c_d_e = q.submit([&](sycl::handler& h) {
        myAccessor acc_c(c, h, sycl::access::mode::read);
        myAccessor acc_d(d, h, sycl::access::mode::read);
        myAccessor acc_e(e, h, sycl::access::mode::write);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            acc_e[idx] = acc_c[idx] + acc_d[idx];
        });
    });
    c.add_event(c_d_e, "c_d_e");
    d.add_event(c_d_e, "c_d_e");
    e.add_event(c_d_e, "c_d_e");

    std::cout << std::endl << " - c_f_g" << std::endl;

    sycl::event c_f_g = q.submit([&](sycl::handler& h) {
        myAccessor acc_c(c, h, sycl::access::mode::read);
        myAccessor acc_f(f, h, sycl::access::mode::read);
        myAccessor acc_g(g, h, sycl::access::mode::write);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            acc_g[idx] = acc_c[idx] + acc_f[idx];
        });
    });
    c.add_event(c_f_g, "c_f_g");
    f.add_event(c_f_g, "c_f_g");
    g.add_event(c_f_g, "c_f_g");

    std::cout << std::endl << " - e_g_h" << std::endl;

    sycl::event e_g_h = q.submit([&](sycl::handler& hdl) {
        myAccessor acc_e(e, hdl, sycl::access::mode::read);
        myAccessor acc_g(g, hdl, sycl::access::mode::read);
        myAccessor acc_h(h, hdl, sycl::access::mode::write);
        hdl.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            acc_h[idx] = acc_e[idx] + acc_g[idx];
        });
    });
    e.add_event(e_g_h, "e_g_h");
    g.add_event(e_g_h, "e_g_h");
    h.add_event(e_g_h, "e_g_h");

    std::cout << std::endl;

    q.wait();

    {
        h.copy_device_to_host();
        myHostAccessor host_acc_h(h, sycl::access::mode::read);
        long long count = 0;
        for (size_t i = 0; i < N; ++i) {
            count += host_acc_h[i];
        }
        std::cout << count << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "test1: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

}

void test2() {

    auto start = std::chrono::high_resolution_clock::now();

    sycl::queue q;

    sycl::buffer<int> a(N);
    sycl::buffer<int> b(N);
    sycl::buffer<int> c(N);
    sycl::buffer<int> d(N);
    sycl::buffer<int> e(N);
    sycl::buffer<int> f(N);
    sycl::buffer<int> g(N);
    sycl::buffer<int> h(N);

    {
        sycl::host_accessor host_acc_a(a, sycl::write_only);
        sycl::host_accessor host_acc_b(b, sycl::write_only);
        sycl::host_accessor host_acc_d(d, sycl::write_only);
        sycl::host_accessor host_acc_f(f, sycl::write_only);
        for (size_t i = 0; i < N; ++i) {
            host_acc_a[i] = i;
            host_acc_b[i] = i;
            host_acc_d[i] = i;
            host_acc_f[i] = i;
        }
    }

    q.submit([&](sycl::handler& h) {
        sycl::accessor acc_a(a, h, sycl::read_only);
        sycl::accessor acc_b(b, h, sycl::read_only);
        sycl::accessor acc_c(c, h, sycl::write_only);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            acc_c[idx] = acc_a[idx] + acc_b[idx];
        });
    });

    q.submit([&](sycl::handler& h) {
        sycl::accessor acc_c(c, h, sycl::read_only);
        sycl::accessor acc_d(d, h, sycl::read_only);
        sycl::accessor acc_e(e, h, sycl::write_only);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            acc_e[idx] = acc_c[idx] + acc_d[idx];
        });
    });

    q.submit([&](sycl::handler& h) {
        sycl::accessor acc_c(c, h, sycl::read_only);
        sycl::accessor acc_f(f, h, sycl::read_only);
        sycl::accessor acc_g(g, h, sycl::write_only);
        h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            acc_g[idx] = acc_c[idx] + acc_f[idx];
        });
    });

    q.submit([&](sycl::handler& hdl) {
        sycl::accessor acc_e(e, hdl, sycl::read_only);
        sycl::accessor acc_g(g, hdl, sycl::read_only);
        sycl::accessor acc_h(h, hdl, sycl::write_only);
        hdl.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
            acc_h[idx] = acc_e[idx] + acc_g[idx];
        });
    });

    {
        sycl::host_accessor host_acc_h(h, sycl::read_only);
        long long count = 0;
        for (size_t i = 0; i < N; ++i) {
            count += host_acc_h[i];
        }
        std::cout << count << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "test2: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

}

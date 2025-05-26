#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "myBuffer.cpp"
#include "myHostAccessor.cpp"
#include "myAccessor.cpp"

using namespace sycl;

void test1();
void test2();

constexpr size_t N = 10000000;

int main() {
    test1();
    test2();
}

void test1() {

    auto start = std::chrono::high_resolution_clock::now();

    queue q;

    myBuffer<int> a(q, N, "a");
    myBuffer<int> b(q, N, "b");
    myBuffer<int> c(q, N, "c");
    myBuffer<int> d(q, N, "d");
    myBuffer<int> e(q, N, "e");
    myBuffer<int> f(q, N, "f");
    myBuffer<int> g(q, N, "g");
    myBuffer<int> h(q, N, "h");

    {
        myHostAccessor host_acc_a(a, access::mode::write);
        myHostAccessor host_acc_b(b, access::mode::write);
        myHostAccessor host_acc_d(d, access::mode::write);
        myHostAccessor host_acc_f(f, access::mode::write);
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

    event a_b_c = q.submit([&](handler& h) {
        myAccessor acc_a(a, h, access::mode::read);
        myAccessor acc_b(b, h, access::mode::read);
        myAccessor acc_c(c, h, access::mode::write);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc_c[idx] = acc_a[idx] + acc_b[idx];
        });
    });
    a.add_event(a_b_c, "a_b_c");
    b.add_event(a_b_c, "a_b_c");
    c.add_event(a_b_c, "a_b_c");

    event c_d_e = q.submit([&](handler& h) {
        myAccessor acc_c(c, h, access::mode::read);
        myAccessor acc_d(d, h, access::mode::read);
        myAccessor acc_e(e, h, access::mode::write);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc_e[idx] = acc_c[idx] + acc_d[idx];
        });
    });
    c.add_event(c_d_e, "c_d_e");
    d.add_event(c_d_e, "c_d_e");
    e.add_event(c_d_e, "c_d_e");

    event c_f_g = q.submit([&](handler& h) {
        myAccessor acc_c(c, h, access::mode::read);
        myAccessor acc_f(f, h, access::mode::read);
        myAccessor acc_g(g, h, access::mode::write);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc_g[idx] = acc_c[idx] + acc_f[idx];
        });
    });
    c.add_event(c_f_g, "c_f_g");
    f.add_event(c_f_g, "c_f_g");
    g.add_event(c_f_g, "c_f_g");

    event e_g_h = q.submit([&](handler& hdl) {
        myAccessor acc_e(e, hdl, access::mode::read);
        myAccessor acc_g(g, hdl, access::mode::read);
        myAccessor acc_h(h, hdl, access::mode::write);
        hdl.parallel_for(range<1>(N), [=](id<1> idx) {
            acc_h[idx] = acc_e[idx] + acc_g[idx];
        });
    });
    e.add_event(e_g_h, "e_g_h");
    g.add_event(e_g_h, "e_g_h");
    h.add_event(e_g_h, "e_g_h");

    {
        h.copy_device_to_host();
        myHostAccessor host_acc_h(h, access::mode::read);
        long long count = 0;
        for (size_t i = 0; i < N; ++i) {
            count += host_acc_h[i];
        }
        std::cout << count << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "test1: Durata: " << duration.count() << " millisecondi" << std::endl;

}

void test2() {

    auto start = std::chrono::high_resolution_clock::now();

    queue q;

    buffer<int> a(N);
    buffer<int> b(N);
    buffer<int> c(N);
    buffer<int> d(N);
    buffer<int> e(N);
    buffer<int> f(N);
    buffer<int> g(N);
    buffer<int> h(N);

    {
        host_accessor host_acc_a(a, write_only);
        host_accessor host_acc_b(b, write_only);
        host_accessor host_acc_d(d, write_only);
        host_accessor host_acc_f(f, write_only);
        for (size_t i = 0; i < N; ++i) {
            host_acc_a[i] = i;
            host_acc_b[i] = i;
            host_acc_d[i] = i;
            host_acc_f[i] = i;
        }
    }

    q.submit([&](handler& h) {
        accessor acc_a(a, h, read_only);
        accessor acc_b(b, h, read_only);
        accessor acc_c(c, h, write_only);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc_c[idx] = acc_a[idx] + acc_b[idx];
        });
    });

    q.submit([&](handler& h) {
        accessor acc_c(c, h, read_only);
        accessor acc_d(d, h, read_only);
        accessor acc_e(e, h, write_only);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc_e[idx] = acc_c[idx] + acc_d[idx];
        });
    });

    q.submit([&](handler& h) {
        accessor acc_c(c, h, read_only);
        accessor acc_f(f, h, read_only);
        accessor acc_g(g, h, write_only);
        h.parallel_for(range<1>(N), [=](id<1> idx) {
            acc_g[idx] = acc_c[idx] + acc_f[idx];
        });
    });

    q.submit([&](handler& hdl) {
        accessor acc_e(e, hdl, read_only);
        accessor acc_g(g, hdl, read_only);
        accessor acc_h(h, hdl, write_only);
        hdl.parallel_for(range<1>(N), [=](id<1> idx) {
            acc_h[idx] = acc_e[idx] + acc_g[idx];
        });
    });

    {
        host_accessor host_acc_h(h, read_only);
        long long count = 0;
        for (size_t i = 0; i < N; ++i) {
            count += host_acc_h[i];
        }
        std::cout << count << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "test2: Durata: " << duration.count() << " millisecondi" << std::endl;

}

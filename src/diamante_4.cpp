#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "sycl_usm/sycl_usm.hpp"

namespace diamante_4 {

    void test1();
    void test2();
    void test3();

    int N;

    void diamante_4(int size) {
        N = size;
        test1();
        test2();
        test3();
    }

    void test1() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        {

            sycl_usm::buffer<int> a(q, N, "a");
            sycl_usm::buffer<int> b(q, N, "b");
            sycl_usm::buffer<int> c(q, N, "c");
            sycl_usm::buffer<int> d(q, N, "d");
            sycl_usm::buffer<int> e(q, N, "e");
            sycl_usm::buffer<int> f(q, N, "f");
            sycl_usm::buffer<int> g(q, N, "g");
            sycl_usm::buffer<int> h(q, N, "h");

            {
                sycl_usm::host_accessor host_acc_a(a, sycl::access::mode::write);
                sycl_usm::host_accessor host_acc_b(b, sycl::access::mode::write);
                sycl_usm::host_accessor host_acc_d(d, sycl::access::mode::write);
                sycl_usm::host_accessor host_acc_f(f, sycl::access::mode::write);
                for (size_t i = 0; i < N; ++i) {
                    host_acc_a[i] = i;
                    host_acc_b[i] = i;
                    host_acc_d[i] = i;
                    host_acc_f[i] = i;
                }
            }

            //std::cout << std::endl << " - a_b_c" << std::endl;

            a.prepareForDevice();
            b.prepareForDevice();
            c.prepareForDevice();
            sycl::event a_b_c = q.submit([&](sycl::handler& h) {
                sycl_usm::accessor acc_a(a, h, sycl::access::mode::read);
                sycl_usm::accessor acc_b(b, h, sycl::access::mode::read);
                sycl_usm::accessor acc_c(c, h, sycl::access::mode::write);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    acc_c[idx] = acc_a[idx] + acc_b[idx];
                    });
                });
            a.add_event(a_b_c, "a_b_c");
            b.add_event(a_b_c, "a_b_c");
            c.add_event(a_b_c, "a_b_c");

            //std::cout << std::endl << " - c_d_e" << std::endl;

            c.prepareForDevice();
            d.prepareForDevice();
            e.prepareForDevice();
            sycl::event c_d_e = q.submit([&](sycl::handler& h) {
                sycl_usm::accessor acc_c(c, h, sycl::access::mode::read);
                sycl_usm::accessor acc_d(d, h, sycl::access::mode::read);
                sycl_usm::accessor acc_e(e, h, sycl::access::mode::write);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    acc_e[idx] = acc_c[idx] + acc_d[idx];
                    });
                });
            c.add_event(c_d_e, "c_d_e");
            d.add_event(c_d_e, "c_d_e");
            e.add_event(c_d_e, "c_d_e");

            //std::cout << std::endl << " - c_f_g" << std::endl;

            c.prepareForDevice();
            f.prepareForDevice();
            g.prepareForDevice();
            sycl::event c_f_g = q.submit([&](sycl::handler& h) {
                sycl_usm::accessor acc_c(c, h, sycl::access::mode::read);
                sycl_usm::accessor acc_f(f, h, sycl::access::mode::read);
                sycl_usm::accessor acc_g(g, h, sycl::access::mode::write);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    acc_g[idx] = acc_c[idx] + acc_f[idx];
                    });
                });
            c.add_event(c_f_g, "c_f_g");
            f.add_event(c_f_g, "c_f_g");
            g.add_event(c_f_g, "c_f_g");

            //std::cout << std::endl << " - e_g_h" << std::endl;

            e.prepareForDevice();
            g.prepareForDevice();
            h.prepareForDevice();
            sycl::event e_g_h = q.submit([&](sycl::handler& hdl) {
                sycl_usm::accessor acc_e(e, hdl, sycl::access::mode::read);
                sycl_usm::accessor acc_g(g, hdl, sycl::access::mode::read);
                sycl_usm::accessor acc_h(h, hdl, sycl::access::mode::write);
                hdl.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    acc_h[idx] = acc_e[idx] + acc_g[idx];
                    });
                });
            e.add_event(e_g_h, "e_g_h");
            g.add_event(e_g_h, "e_g_h");
            h.add_event(e_g_h, "e_g_h");

            //std::cout << std::endl;

            {
                sycl_usm::host_accessor host_acc_h(h, sycl::access::mode::read);
                long long count = 0;
                for (size_t i = 0; i < N; ++i) {
                    count += host_acc_h[i];
                }
                std::cout << count << std::endl;
            }

        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "test1 mySycl: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

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
        std::cout << "test2 sycl: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

    }

    void test3() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        int* a = sycl::malloc_device<int>(N, q);
        int* b = sycl::malloc_device<int>(N, q);
        int* c = sycl::malloc_device<int>(N, q);
        int* d = sycl::malloc_device<int>(N, q);
        int* e = sycl::malloc_device<int>(N, q);
        int* f = sycl::malloc_device<int>(N, q);
        int* g = sycl::malloc_device<int>(N, q);
        int* h = sycl::malloc_device<int>(N, q);

        std::vector<int> host_a(N);
        std::vector<int> host_b(N);
        std::vector<int> host_d(N);
        std::vector<int> host_f(N);

        std::vector<int> host_h(N);

        //inizializzo dati host
        for (size_t i = 0; i < N; ++i) {
            host_a[i] = i;
            host_b[i] = i;
            host_d[i] = i;
            host_f[i] = i;
        }

        //copio dati da host a device
        q.memcpy(a, host_a.data(), N * sizeof(int)).wait();
        q.memcpy(b, host_b.data(), N * sizeof(int)).wait();
        q.memcpy(d, host_d.data(), N * sizeof(int)).wait();
        q.memcpy(f, host_f.data(), N * sizeof(int)).wait();

        //c = a + b
        auto cab = q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                c[idx] = a[idx] + b[idx];
                });
            });

        //e = c + d
        auto ecd = q.submit([&](sycl::handler& h) {
            h.depends_on(cab);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                e[idx] = c[idx] + d[idx];
                });
            });

        //g = c + f
        auto gcf = q.submit([&](sycl::handler& h) {
            h.depends_on(cab);
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                g[idx] = c[idx] + f[idx];
                });
            });

        //h = e + g
        q.submit([&](sycl::handler& hdl) {
            hdl.depends_on(ecd);
            hdl.depends_on(gcf);
            hdl.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                h[idx] = e[idx] + g[idx];
                });
            }).wait();

        //copio dati da device a host
        q.memcpy(host_h.data(), h, N * sizeof(int)).wait();

        long long count = 0;
        for (size_t i = 0; i < N; ++i) {
            count += host_h[i];
        }
        std::cout << count << std::endl;

        sycl::free(a, q);
        sycl::free(b, q);
        sycl::free(c, q);
        sycl::free(d, q);
        sycl::free(e, q);
        sycl::free(f, q);
        sycl::free(g, q);
        sycl::free(h, q);

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "test3 esplicito: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

    }

}
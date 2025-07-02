#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "mysycl/mysycl.hpp"
#include <vector>

namespace dipendenza_albero_binario {

    void test1();
    void test2();
    void test3();

    int N;
    int num_kernels;
    int livelli;
    int num_foglie;

    void dipendenza_albero_binario(int size, int nk) {
        N = size;
        num_kernels = nk;
        livelli = __builtin_ctz(nk);
        num_foglie = 1 << (livelli - 1);
        std::cout << "Num_foglie: " << num_foglie << std::endl << std::endl;
        test1();
        //test2();
        //test3();
    }

    void test1() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        {

            std::vector<mysycl::buffer<int>> buffers;
            buffers.reserve(num_kernels - 1);
            for (int i = 0; i < num_kernels - 1; i++) {
                buffers.emplace_back(q, N);
            }

            //kernel 0
            buffers[0].add_event(q.submit([&](sycl::handler& h) {
                mysycl::accessor acc(buffers[0], h, sycl::access::mode::write);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    acc[idx] = 1;
                    });
                }), "");

            //kernel da 1 a 'num_kernels - 2'
            for (int i = 1; i < num_kernels - 1; i++) {
                int padre_idx = (i-1)/2;
                std::cout << "figlio: " << i << " padre: " << padre_idx << std::endl;
                auto e = q.submit([&](sycl::handler& h) {
                    mysycl::accessor acc(buffers[i], h, sycl::access::mode::write);
                    mysycl::accessor padre(buffers[padre_idx], h, sycl::access::mode::read);
                    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                        acc[idx] = padre[idx];
                        });
                    });
                buffers[i].add_event(e, "");
                buffers[padre_idx].add_event(e, "");
            }

            q.wait();

            long long count = 0;
            for (int i = 0; i < num_foglie; i++) {
                int indice_foglia = num_foglie - 1 + i;
                std::cout << "indice_foglia: " << indice_foglia << "; ";
                buffers[indice_foglia].copy_device_to_host();
                mysycl::host_accessor host_acc(buffers[indice_foglia], sycl::access::mode::read);
                count += host_acc[0];
                std::cout << "count: " << count << std::endl;
            }

        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "test1 mySycl: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

    }

    void test2() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        {

            sycl::buffer<int> buffer(N);

            for (int i = 0; i < num_kernels; i++) {
                q.submit([&](sycl::handler& h) {
                    sycl::accessor acc(buffer, h, sycl::write_only);
                    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                        acc[idx] = i + 1;
                        });
                    });
            }

            sycl::host_accessor host_acc(buffer, sycl::read_only);
            long long count = 0;
            for (int j = 0; j < N; j++) {
                count += host_acc[j];
            }
            std::cout << count / N << std::endl;

        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "test2 sycl: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

    }

    void test3() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        int* array_dev = sycl::malloc_device<int>(N, q);
        int* array_host = new int[N];

        sycl::event e;

        //kernel 0
        e = q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                array_dev[idx] = 0 + 1;
                });
            });

        //kernel da 1 a 'num_kernels - 1'
        for (int i = 1; i < num_kernels; i++) {
            e = q.submit([&](sycl::handler& h) {
                h.depends_on(e);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    array_dev[idx] = i + 1;
                });
            });
        }

        q.wait();

        q.memcpy(array_host, array_dev, N * sizeof(int)).wait();
        long long count = 0;
        for (int j = 0; j < N; j++) {
            count += array_host[j];
        }
        std::cout << count / N << std::endl;

        sycl::free(array_dev, q);
        delete[] array_host;

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "test3 esplicito: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

    }

}
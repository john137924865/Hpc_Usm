#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "sycl_usm/sycl_usm.hpp"
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
        test2();
        test3();
    }

    void test1() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        {

            std::vector<sycl_usm::buffer<int>> buffers;
            buffers.reserve(num_kernels - 1);
            for (int i = 0; i < num_kernels - 1; i++) {
                buffers.emplace_back(q, N);
            }

            //kernel 0
            buffers[0].prepareForDevice();
            buffers[0].add_event(q.submit([&](sycl::handler& h) {
                sycl_usm::accessor acc(buffers[0], h, sycl::access::mode::write);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    acc[idx] = 1;
                    });
                }), "");

            //kernel da 1 a 'num_kernels - 2'
            for (int i = 1; i < num_kernels - 1; i++) {
                int padre_idx = (i-1)/2;
                buffers[i].prepareForDevice();
                buffers[padre_idx].prepareForDevice();
                //std::cout << "figlio: " << i << " padre: " << padre_idx << std::endl;
                auto e = q.submit([&](sycl::handler& h) {
                    sycl_usm::accessor acc(buffers[i], h, sycl::access::mode::write);
                    sycl_usm::accessor padre(buffers[padre_idx], h, sycl::access::mode::read);
                    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                        acc[idx] = padre[idx];
                        });
                    });
                buffers[i].add_event(e, "");
                buffers[padre_idx].add_event(e, "");
            }

            long long count = 0;
            for (int i = 0; i < num_foglie; i++) {
                int indice_foglia = num_foglie - 1 + i;
                //std::cout << "indice_foglia: " << indice_foglia << "; ";
                sycl_usm::host_accessor host_acc(buffers[indice_foglia], sycl::access::mode::read);
                count += host_acc[0];
            }
            std::cout << "count: " << count << std::endl;

        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "test1 mySycl: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

    }

    void test2() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        {

            std::vector<sycl::buffer<int>> buffers;
            buffers.reserve(num_kernels - 1);
            for (int i = 0; i < num_kernels - 1; i++) {
                buffers.emplace_back(N);
            }

            //kernel 0
            q.submit([&](sycl::handler& h) {
                sycl::accessor acc(buffers[0], h, sycl::write_only);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    acc[idx] = 1;
                    });
                });

            //kernel da 1 a 'num_kernels - 2'
            for (int i = 1; i < num_kernels - 1; i++) {
                int padre_idx = (i - 1) / 2;
                //std::cout << "figlio: " << i << " padre: " << padre_idx << std::endl;
                q.submit([&](sycl::handler& h) {
                    sycl::accessor acc(buffers[i], h, sycl::write_only);
                    sycl::accessor padre(buffers[padre_idx], h, sycl::read_only);
                    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                        acc[idx] = padre[idx];
                        });
                    });
            }

            long long count = 0;
            for (int i = 0; i < num_foglie; i++) {
                int indice_foglia = num_foglie - 1 + i;
                //std::cout << "indice_foglia: " << indice_foglia << "; ";
                sycl::host_accessor host_acc(buffers[indice_foglia], sycl::read_only);
                count += host_acc[0];
            }
            std::cout << "count: " << count << std::endl;

        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "test2 sycl: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

    }

    void test3() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        std::vector<int*> arrays_dev;
        std::vector<int*> arrays_host;
        std::vector<sycl::event> events;
        arrays_dev.reserve(num_kernels - 1);
        arrays_host.reserve(num_foglie);
        events.reserve(num_kernels - 1);
        for (int i = 0; i < num_kernels - 1; i++) {
            arrays_dev[i] = sycl::malloc_device<int>(N, q);
        }
        for (int i = 0; i < num_foglie; i++) {
            arrays_host[i] = new int[N];
        }

        //kernel 0
        int* array_dev_0 = arrays_dev[0];
        events.push_back(q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                array_dev_0[idx] = 1;
                });
            }));

        //kernel da 1 a 'num_kernels - 2'
        for (int i = 1; i < num_kernels - 1; i++) {
            int padre_idx = (i - 1) / 2;
            int* figlio = arrays_dev[i];
            int* padre = arrays_dev[padre_idx];
            auto e = events[padre_idx];
            //std::cout << "figlio: " << i << " padre: " << padre_idx << std::endl;
            events.push_back(q.submit([&](sycl::handler& h) {
                h.depends_on(e);
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    figlio[idx] = padre[idx];
                    });
                }));
        }

        q.wait();

        long long count = 0;
        for (int i = 0; i < num_foglie; i++) {
            int indice_foglia = num_foglie - 1 + i;
            //std::cout << "indice_foglia: " << indice_foglia << "; ";
            q.memcpy(arrays_host[i], arrays_dev[indice_foglia], N * sizeof(int)).wait();
            count += arrays_host[i][0];
        }
        std::cout << "count: " << count << std::endl;

        for (int i = 0; i < num_kernels - 1; i++) {
            sycl::free(arrays_dev[i], q);
        }
        for (int i = 0; i < num_foglie; i++) {
            delete[] arrays_host[i];
        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "test3 esplicito: Durata: " << duration.count() << " millisecondi" << std::endl << std::endl;

    }

}
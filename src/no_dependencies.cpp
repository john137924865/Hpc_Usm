#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "buff_acc_lib/buff_acc_lib.hpp"
#include <vector>

namespace no_dependencies {

    void test1();
    void test2();
    void test3();

    int N;
    int num_kernels;

    void no_dependencies(int size, int nk) {
        N = size;
        num_kernels = nk;
        test1();
        test2();
        test3();
    }

    void test1() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        {

            std::vector<buff_acc_lib::buffer<int>> buffers;
            buffers.reserve(num_kernels);
            for (int i = 0; i < num_kernels; i++) {
                buffers.emplace_back(q, N);
            }

            for (int i = 0; i < num_kernels; i++) {
                buffers[i].prepareForDevice();
                buffers[i].add_event(q.submit([&](sycl::handler& h) {
                    buff_acc_lib::accessor acc(buffers[i], h, sycl::access::mode::write);
                    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                        acc[idx] = idx;
                        });
                    }), "");
            }
 
            for (int i = 0; i < num_kernels; i++) {
                buff_acc_lib::host_accessor host_acc(buffers[i], sycl::access::mode::read);
                long long count = 0;
                for (int j = 0; j < N; j++) {
                    count += host_acc[j];
                }
                //std::cout << count << std::endl;
            }

        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "buff_acc_lib: " << duration.count() << " ms" << std::endl << std::endl;

    }

    void test2() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        {

            std::vector<sycl::buffer<int>> buffers;
            buffers.reserve(num_kernels);
            for (int i = 0; i < num_kernels; i++) {
                buffers.emplace_back(N);
            }

            for (int i = 0; i < num_kernels; i++) {
                q.submit([&](sycl::handler& h) {
                    sycl::accessor acc(buffers[i], h, sycl::write_only);
                    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                        acc[idx] = idx;
                        });
                    });
            }

            for (int i = 0; i < num_kernels; i++) {
                sycl::host_accessor host_acc(buffers[i], sycl::read_only);
                long long count = 0;
                for (int j = 0; j < N; j++) {
                    count += host_acc[j];
                }
                //std::cout << count << std::endl;
            }

        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Sycl buffer-accessor: " << duration.count() << " ms" << std::endl << std::endl;

    }

    void test3() {

        auto start = std::chrono::high_resolution_clock::now();

        sycl::queue q;

        std::vector<int*> arrays_dev(num_kernels);
        std::vector<int*> arrays_host(num_kernels);
        for (int i = 0; i < num_kernels; i++) {
            arrays_dev[i] = sycl::malloc_device<int>(N, q);
            arrays_host[i] = new int[N];
        }

        for (int i = 0; i < num_kernels; i++) {
            int* arr_dev = arrays_dev[i];
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                    arr_dev[idx] = idx;
                });
            });
        }

        q.wait();

        for (int i = 0; i < num_kernels; i++) {
            q.memcpy(arrays_host[i], arrays_dev[i], N * sizeof(int)).wait();
            long long count = 0;
            for (int j = 0; j < N; j++) {
                count += arrays_host[i][j];
            }
            //std::cout << count << std::endl;
        }

        for (int i = 0; i < num_kernels; i++) {
            sycl::free(arrays_dev[i], q);
            delete[] arrays_host[i];
        }

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "USM device allocation: " << duration.count() << " ms" << std::endl << std::endl;

    }

}
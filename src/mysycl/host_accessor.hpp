#pragma once

#include <sycl/sycl.hpp>
#include <iostream>

namespace mysycl {

    template <typename T>
    class host_accessor {

    private:
        T* host_data;
        sycl::access::mode mode;

    public:

        host_accessor(mysycl::buffer<T>& buf, sycl::access::mode mode = sycl::access::mode::read_write) :
            host_data(buf.get_host_data()), mode(mode) {
                /* data movement */
                buf.get_queue()->wait();
                //se sta su device copio su host
                if (buf.getLastData() == 0) {
                    std::cout << "copio device to host " << std::endl;
                    buf.copy_device_to_host();
                    //se solo lettura metto 2 both altrimenti 1 host
                    if (mode == sycl::access::mode::read) {
                        buf.setLastData(2);
                    } else {
                        buf.setLastData(1);
                    }
                }
                //se sta su entrambi controllo se va su host
                else if (buf.getLastData() == 2) {
                    if (mode != sycl::access::mode::read) {
                        buf.setLastData(1);
                    }
                }
        }

        T& operator[](size_t index) {
            return host_data[index];
        }

    };
}
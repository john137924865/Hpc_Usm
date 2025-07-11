#pragma once

#include <sycl/sycl.hpp>
#include <iostream>

namespace mysycl {

    template <typename T>
    class accessor {

    private:
        T* device_data;
        sycl::access::mode mode;

    public:

        accessor(mysycl::buffer<T>& buf, sycl::handler& h, sycl::access::mode mode = sycl::access::mode::read_write) :
            device_data(buf.get_device_data()), mode(mode) {
                /* dipendenze eventi */
                buf.check_mode(mode, h);

                /* data movement */
                //se sta su host (ho già copiato in prepareForDevice())
                if (buf.getLastData() == 1) {
                    //se solo lettura metto 2 both altrimenti 0 device
                    if (mode == sycl::access::mode::read) {
                        buf.setLastData(2);
                    } else {
                        buf.setLastData(0);
                    }
                }
                //se sta su entrambi controllo se va su device
                else if (buf.getLastData() == 2) {
                    if (mode != sycl::access::mode::read) {
                        buf.setLastData(0);
                    }
                }
        }

        T& operator[](size_t index) const {
            return device_data[index];
        }

    };
}
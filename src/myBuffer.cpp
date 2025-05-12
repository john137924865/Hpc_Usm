#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

using namespace sycl;

template <typename T>
class myBuffer {

    private:
        T* device_data;
        T* host_data;
        size_t size;
        queue* q;
        std::vector<event> events;
        bool write = false;

    public:

        myBuffer(queue& q, size_t n) : size(n) {
            this->device_data = static_cast<T*>(malloc_device(sizeof(T) * size, q));
            this->host_data = new T[size];
            this->q = &q;
        }

        void copy_host_to_device() {
            (*q).memcpy(device_data, host_data, sizeof(T) * size).wait();
        }

        void copy_device_to_host() {
            (*q).memcpy(host_data, device_data, sizeof(T) * size).wait();
        }

        T* get_device_data() {
            return device_data;
        }
        
        T* get_host_data() {
            return host_data;
        }

        void add_event(event e) {
            if (write) {
                e.wait();
                write = false;
            } else {
                events.push_back(e);
            }
            std::cout << "add_event; size : " << events.size() << " write: " << std::boolalpha << write << std::endl << std::endl;
        }

        void check_mode(access::mode mode, handler& h) {
            if (mode != access::mode::read) {
                for (event& elem : events) {
                    h.depends_on(elem);
                }
                events.clear();
                write = true;
            }
            std::cout << "check_mode; size : " << events.size() << " write: " << std::boolalpha << write << std::endl;
        }

        /*~myBuffer() {
            std::cout << events.size() << std::endl << modes.size() << std::endl;
        }*/

};
#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>

namespace mysycl {

    template <typename T>
    class buffer {

    private:
        std::string name;
        T* device_data = nullptr;
        T* host_data = nullptr;
        size_t size;
        sycl::queue* q = nullptr;
        std::vector<sycl::event> events;
        std::vector<std::string> names;
        std::vector<sycl::event> last_write;
        std::vector<std::string> last_write_name;
        bool current_write = false;
        bool ever_write = false;

    public:

        buffer(sycl::queue& q, size_t n, std::string name = "") : size(n) {
            this->device_data = static_cast<T*>(malloc_device(sizeof(T) * size, q));
            this->host_data = new T[size];
            this->q = &q;
            this->name = name;
        }

        ~buffer() {
            (*q).wait();
            sycl::free(device_data, *q);
            delete[] host_data;
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

        sycl::queue* get_queue() {
            return this->q;
        }

        void check_mode(sycl::access::mode mode, sycl::handler& h) {
            if (mode != sycl::access::mode::read) {
                //std::cout << name << " depends_on " << events.size() << " ";
                for (int i = 0; i < events.size(); i++) {
                    h.depends_on(events[i]);
                    //std::cout << names[i] << " ";
                }
                //std::cout << ";" << std::endl;
                events.clear();
                names.clear();
                current_write = true;
            }
            if (ever_write) {
                h.depends_on(last_write[0]);
                //std::cout << name << " depends_on last_write " << last_write_name[0] << std::endl;
            }
            //std::cout << "check_mode; size : " << events.size() << ", write: " << std::boolalpha << write << std::endl;
        }

        void add_event(sycl::event e, std::string name) {
            if (current_write) {
                ever_write = true;
                last_write.clear();
                last_write_name.clear();
                last_write.push_back(e);
                last_write_name.push_back(name);
                //std::cout << this->name << " aggiorno last_write con " << name << std::endl;
                current_write = false;
            }
            else {
                events.push_back(e);
                names.push_back(name);
                //std::cout << this->name << " push_back events e names " << name << std::endl;
            }
            //std::cout << "add_event; size : " << events.size() << ", write: " << std::boolalpha << write << std::endl;
        }

    };
}
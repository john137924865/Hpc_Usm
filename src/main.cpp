#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "mysycl/mysycl.hpp"
#include "diamante_4.cpp"
#include "no_dipendenze.cpp"

int main() {
    int size, num_kernels;
    std::cout << "Milioni: ";
    std::cin >> size;
    std::cout << size << std::endl << std::endl;
    std::cout << "Numero kernel: ";
    std::cin >> num_kernels;
    std::cout << num_kernels << std::endl << std::endl;
    size *= 1000000;

    no_dipendenze::no_dipendenze(size, num_kernels);
}
#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include "diamond_4.cpp"
#include "no_dependencies.cpp"
#include "linear_dependencies.cpp"
#include "binary_tree_dependencies.cpp"

int main() {
    int size, num_kernels, choice;

    std::cout << "Migliaia: ";
    std::cin >> size;
    size *= 1000;
    std::cout << size << std::endl << std::endl;

    std::cout << "Choose the function to execute:\n"
        << "1. diamond_4\n"
        << "2. no dependencies\n"
        << "3. linear dependencies\n"
        << "4. binary tree dependencies\n"
        << "Choice : ";
    std::cin >> choice;
    std::cout << std::endl;

    switch (choice) {
    case 1:
        diamond_4::diamond_4(size);
        break;
    case 2:
    case 3:
    case 4:
        std::cout << "Number of kernels: ";
        std::cin >> num_kernels;
        std::cout << num_kernels << std::endl << std::endl;

        if (choice == 2)
            no_dependencies::no_dependencies(size, num_kernels);
        else if (choice == 3)
            linear_dependencies::linear_dependencies(size, num_kernels);
        else
            binary_tree_dependencies::binary_tree_dependencies(size, num_kernels);

        break;
    default:
        std::cout << "Invalid choice.\n";
        break;
    }
}
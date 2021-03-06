#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <numeric>

#include <stdio.h>
#include <stdlib.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

using namespace std;

//We may change this value!!
const int FILTER_WIDTH = 7;

//We may change this value!!!
int FILTER[FILTER_WIDTH*FILTER_WIDTH] = {
	1,4,7,10,7,4,1,
	4,12,26,33,26,12,4,
	7,26,55,71,55,26,7,
	10,33,71,91,71,33,10,
	7,26,55,71,55,26,7,
	4,12,26,33,26,12,4,
	1,4,7,10,7,4,1
};

// Display the first and last 10 items
// For debug only
void displayResult(const int original[], const int result[], int size) {
	cout << "Display result: ";
	cout << "(original -> result)\n";

	for (int i = 0; i < 10; i++) {
		cout << original[i] << " -> " << result[i] << "\n";
	}
	cout << ".\n.\n.\n";

	for (int i = size - 10; i < size; i++) {
		cout << original[i] << " -> " << result[i] << "\n";
	}
}

void initColorData(string file, int **data, int *sizeX, int *sizeY) {
	int x;
	int y;
	long long i = 0;
	cout << "Reading "<< file << "... \n";
	ifstream myfile(file);
	if (myfile.is_open()) {
		myfile >> x;
		myfile >> y;

		int *temp = new int[x * y * 3];
		for( i=0; i < x * y * 3; i++){
			myfile >> temp[(int)i];
		}
		myfile.close();
		*data = temp;
		*sizeX = x;
		*sizeY = y;
	}
	else {
		cout << "ERROR: File " << file << " not found!\n";
		exit(0);
	}
	cout << i << " entries imported\n";
}

void saveResult(string file, int data[], int sizeX, int sizeY) {
	long long i = 0;
	cout << "Saving data to "<< file <<"... \n";
	ofstream myfile(file, std::ofstream::out);
	if (myfile.is_open()) {
		myfile << sizeX << "\n";
		myfile << sizeY << "\n";
		for (i = 0; i < sizeX * sizeY; i++){
			myfile << data[3* i] << " " << data[3* i + 1] << " " << data[3* i+ 2]<< "\n";
		}
		myfile.close();
	}
	else {
		cout << "ERROR: Cannot save to " << file << "!\n";
		exit(0);
	}
	cout << i << " entries saved\n";
}

void flipFilter(int *filter, int *result, int filterWidth){
	for (int i=0; i < filterWidth*filterWidth; i++) result[filterWidth*filterWidth-i-1] = filter[i];
}

// TODO: implement the kneral function for 2D smoothing 
__global__ void smoothen(int *data, int *result, int *filter, int sizeX, int sizeY, int FILTER_WIDTH, int filterSum){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if(idx < sizeX * sizeY * 3) {
		int z = idx % 3;
		int x = (idx/3) % sizeX - FILTER_WIDTH/2;
		int y = idx/(sizeX * 3) - FILTER_WIDTH/2;

		
		int value = 0;
		for (int i = y; i < y + FILTER_WIDTH; i++){
			for (int j = x; j < x + FILTER_WIDTH; j++){
				if (i > -1 && i < sizeY && j > -1 && j < sizeX){
					value += filter[(i-y)*FILTER_WIDTH + (j-x)] * data[i*sizeX*3 + j*3 + z];
				}
			}
		}
		if(filterSum != 0) value /= filterSum;
		if (value < 0) value =  0;
		if (value > 255) value = 255;
		result[idx] = value;
	}
}

// GPU implementation
void GPU_Test(int data[], int result[], int sizeX, int sizeY) {
	// input:
	//	int data[] - int array holding the flattened original image
	//	int sizeX - the width of the image
	//	int sizeY - the height of the image
	// output:
	//	int result[] - int array holding the image

	int filterSum = accumulate(begin(FILTER), end(FILTER), 0, plus<int>());
	int *filter = new int[FILTER_WIDTH*FILTER_WIDTH];
	flipFilter(FILTER, filter, FILTER_WIDTH);

	// TODO: allocate device memory and copy data onto the device
	int *d_data, *d_result, *d_filter;
	int size = (sizeX * sizeY * 3) * sizeof(int);
	
	cudaMalloc((void **)&d_data, size);
	cudaMalloc((void **)&d_result, size);
	cudaMalloc((void **)&d_filter, FILTER_WIDTH * FILTER_WIDTH * sizeof(int));

	cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_filter, filter, FILTER_WIDTH * FILTER_WIDTH * sizeof(int), cudaMemcpyHostToDevice);

	// Start timer for kernel
	auto startKernel = chrono::steady_clock::now();
	const int n_blocks = (sizeX * sizeY * 3)/BLOCK_SIZE;

	// TODO: call the kernel function
	smoothen<<<n_blocks, BLOCK_SIZE>>>(d_data, d_result, d_filter, sizeX, sizeY, FILTER_WIDTH, filterSum);

	// End timer for kernel and display kernel time
	cudaDeviceSynchronize(); // <- DO NOT REMOVE
	auto endKernel = chrono::steady_clock::now();
	cout << "Kernel Elapsed time: " << chrono::duration <double, milli>(endKernel - startKernel).count() << "ms\n";

	// TODO: copy reuslt from device to host
	cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

	// TODO: free device memory
	cudaFree(d_data); cudaFree(d_result); cudaFree(d_filter);
}


// CPU implementation
void CPU_Test(int data[], int result[], int sizeX, int sizeY) {
	// input:
	//	int data[] - int array holding the flattened original image
	//	int sizeX - the width of the image
	//	int sizeY - the height of the image
	// output:
	//	int result[] - int array holding the image

	// TODO: smooth the image with filter size = FILTER_WIDTH
	//       apply zero padding for the border
	int filterSum = accumulate(begin(FILTER), end(FILTER), 0, plus<int>());
	int *filter = new int[FILTER_WIDTH*FILTER_WIDTH];
	flipFilter(FILTER, filter, FILTER_WIDTH);

	long long idx = 0;
	for (idx = 0; idx < sizeX * sizeY * 3; idx++){
		int z = idx % 3;
		int x = (idx/3) % sizeX - FILTER_WIDTH/2;
		int y = idx/(sizeX * 3) - FILTER_WIDTH/2;

		
		int value = 0;
		for (int i = y; i < y + FILTER_WIDTH; i++){
			for (int j = x; j < x + FILTER_WIDTH; j++){
				if (i > -1 && i < sizeY && j > -1 && j < sizeX){
					value += filter[(i-y)*FILTER_WIDTH + (j-x)] * data[i*sizeX*3 + j*3 + z];
				}
			}
		}
		
		if(filterSum != 0) value /= filterSum;
		if (value < 0) value =  0;
		if (value > 255) value = 255;
		result[idx] = value;
	}
}

// The image is flattened into a text file of pixel values.
int main(int argc, char *argv[]) {
	string inputFile = (argc == 1) ? "image_color.txt" : argv[1];

	int sizeX;
	int sizeY;
	int *dataForCPUTest;
	int *dataForGPUTest;	

	initColorData(inputFile, &dataForCPUTest, &sizeX, &sizeY);
	initColorData(inputFile, &dataForGPUTest, &sizeX, &sizeY);

	int size = sizeX * sizeY * 3;
	int *resultForCPUTest = new int[size];
	int *resultForGPUTest = new int[size];

	cout << "\n";

	cout << "CPU Implementation\n";

	auto startCPU = chrono::steady_clock::now();
	CPU_Test(dataForCPUTest, resultForCPUTest, sizeX, sizeY);
	auto endCPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endCPU - startCPU).count() << "ms\n";

	// displayResult(dataForCPUTest, resultForCPUTest, size);

	saveResult("color_result_CPU.txt",resultForCPUTest, sizeX, sizeY);

	cout << "\n";
	cout << "GPU Implementation\n";

	auto startGPU = chrono::steady_clock::now();
	GPU_Test(dataForGPUTest, resultForGPUTest, sizeX, sizeY);
	auto endGPU = chrono::steady_clock::now();

	cout << "Elapsed time: " << chrono::duration <double, milli>(endGPU - startGPU).count() << "ms\n";

	// displayResult(dataForGPUTest, resultForGPUTest, size);
	saveResult("color_result_GPU.txt",resultForGPUTest, sizeX, sizeY);

	return 0;
}

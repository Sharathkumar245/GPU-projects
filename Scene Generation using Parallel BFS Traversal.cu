/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

void readFile(const char *fileName, std::vector<SceneNode *> &scenes, std::vector<std::vector<int>> &edges, std::vector<std::vector<int>> &translations, int &frameSizeX, int &frameSizeY)
{
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen(fileName, "r")) == NULL)
	{
		printf("Failed at opening the file %s\n", fileName);
		return;
	}

	// Input the header information.
	int numMeshes;
	fscanf(inputFile, "%d", &numMeshes);
	fscanf(inputFile, "%d %d", &frameSizeX, &frameSizeY);

	// Input all meshes and store them inside a vector.
	int meshX, meshY;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity;
	int *currMesh;
	for (int i = 0; i < numMeshes; i++)
	{
		fscanf(inputFile, "%d %d", &meshX, &meshY);
		fscanf(inputFile, "%d %d", &globalPositionX, &globalPositionY);
		fscanf(inputFile, "%d", &opacity);
		currMesh = (int *)malloc(sizeof(int) * meshX * meshY);
		for (int j = 0; j < meshX; j++)
		{
			for (int k = 0; k < meshY; k++)
			{
				fscanf(inputFile, "%d", &currMesh[j * meshY + k]);
			}
		}
		// Create a Scene out of the mesh.
		SceneNode *scene = new SceneNode(i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity);
		scenes.push_back(scene);
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf(inputFile, "%d", &relations);
	int u, v;
	for (int i = 0; i < relations; i++)
	{
		fscanf(inputFile, "%d %d", &u, &v);
		edges.push_back({u, v});
	}

	// Input all translations.
	int numTranslations;
	fscanf(inputFile, "%d", &numTranslations);
	std::vector<int> command(3, 0);
	for (int i = 0; i < numTranslations; i++)
	{
		fscanf(inputFile, "%d %d %d", &command[0], &command[1], &command[2]);
		translations.push_back(command);
	}
}

void writeFile(const char *outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY)
{
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen(outputFileName, "w")) == NULL)
	{
		printf("Failed while opening output file\n");
	}

	for (int i = 0; i < frameSizeX; i++)
	{
		for (int j = 0; j < frameSizeY; j++)
		{
			fprintf(outputFile, "%d ", hFinalPng[i * frameSizeY + j]);
		}
		fprintf(outputFile, "\n");
	}
}
__global__ void bfs_kernal(int *gpucsr, int *gpuoffset, bool *temp_vertex_list, bool *temp_level_list, bool *gpurear, int V)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < V)
	{
		if (temp_level_list[tid] != false)
		{
			temp_level_list[tid] = false;
			int start = gpuoffset[tid];
			int end = gpuoffset[tid + 1];
			for (int i = start; i < end; i++)
			{
				temp_vertex_list[gpucsr[i]] = true;
				temp_level_list[gpucsr[i]] = true;
				*gpurear = true;
			}
		}
	}
}
__global__ void final_kernal(int *gpuglobalx, int *gpuglobaly, int *gpuframex, int *gpuframey, int *transforms, int *no_translations, int V, int **verticestouched, int *node_size)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < *no_translations)
	{
		int node = transforms[3 * tid + 0];
		int cmd = transforms[3 * tid + 1];
		int moves = transforms[3 * tid + 2];
		int *vertex = verticestouched[node];

		for (int i = 0; i < node_size[node]; i++)
		{
			if (cmd == 0)
			{
				atomicSub(&gpuglobalx[vertex[i]], moves);
			}
			else if (cmd == 1)
			{
				atomicAdd(&gpuglobalx[vertex[i]], moves);
			}
			else if (cmd == 2)
			{
				atomicSub(&gpuglobaly[vertex[i]], moves);
			}
			else
			{
				atomicAdd(&gpuglobaly[vertex[i]], moves);
			}
		}
	}
}
__global__ void final_mesh(int framex, int framey, int globalx, int globaly, int mesh_opac, int *sceneopacity, int *current_mesh, int *gpuscene, int scenex, int sceney, int V)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < (framex * framey))
	{
		int row = tid / framey;
		int col = tid % framey;
		if (globalx + row < 0 || globalx + row >= scenex || globaly + col < 0 || globaly + col >= sceney)
			;
		else
		{
			if (sceneopacity[(globalx + row) * sceney + (globaly + col)] < mesh_opac)
			{
				gpuscene[(globalx + row) * sceney + (globaly + col)] = current_mesh[row * framey + col];
				sceneopacity[(globalx + row) * sceney + (globaly + col)] = mesh_opac;
			}
		}
	}
}
int main(int argc, char **argv)
{

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1];
	int *hFinalPng;

	int frameSizeX, frameSizeY;
	std::vector<SceneNode *> scenes;
	std::vector<std::vector<int>> edges;
	std::vector<std::vector<int>> translations;
	readFile(inputFileName, scenes, edges, translations, frameSizeX, frameSizeY);
	hFinalPng = (int *)malloc(sizeof(int) * frameSizeX * frameSizeY);

	// Make the scene graph from the matrices.
	Renderer *scene = new Renderer(scenes, edges);

	// Basic information.
	int V = scenes.size();
	int E = edges.size();
	int numTranslations = translations.size();

	// Convert the scene graph into a csr.
	scene->make_csr(); // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset();
	int *hCsr = scene->get_h_csr();
	int *hOpacity = scene->get_opacity();					   // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr();					   // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX(); // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY(); // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX();				   // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY();				   // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now();

	// Code begins here.
	// Do not change anything above this comment.
	int *gpuoffset, *gpucsr, *gpuopacity, **gpumesh, *gpuglobalx, *gpuglobaly, *gpuframex, *gpuframey;
	cudaMalloc(&gpuoffset, sizeof(int) * (V + 1));
	cudaMemcpy(gpuoffset, hOffset, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
	cudaMalloc(&gpuopacity, (sizeof(int) * V));
	cudaMemcpy(gpuopacity, hOpacity, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMalloc(&gpucsr, sizeof(int) * (E));
	cudaMemcpy(gpucsr, hCsr, sizeof(int) * (E), cudaMemcpyHostToDevice);
	cudaMalloc(&gpumesh, (sizeof(int *) * V));
	int *transforms;
	transforms = (int *)malloc(sizeof(int) * 3 * numTranslations);
	int k = 0;
	for (int i = 0; i < numTranslations; i++)
	{
		transforms[k++] = translations[i][0];
		transforms[k++] = translations[i][1];
		transforms[k++] = translations[i][2];
	}
	int *gputransforms;
	cudaMalloc(&gputransforms, sizeof(int) * 3 * numTranslations);
	cudaMemcpy(gputransforms, transforms, sizeof(int) * 3 * numTranslations, cudaMemcpyHostToDevice);

	cudaMalloc(&gpuglobalx, (sizeof(int) * V));
	cudaMemcpy(gpuglobalx, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMalloc(&gpuglobaly, (sizeof(int) * V));
	cudaMemcpy(gpuglobaly, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMalloc(&gpuframex, (sizeof(int) * V));
	cudaMemcpy(gpuframex, hFrameSizeX, sizeof(int) * V, cudaMemcpyHostToDevice);
	cudaMalloc(&gpuframey, (sizeof(int) * V));
	cudaMemcpy(gpuframey, hFrameSizeY, sizeof(int) * V, cudaMemcpyHostToDevice);

	int *cpuscene;
	int *cpuopacity;
	cpuscene = (int *)malloc(sizeof(int) * frameSizeX * frameSizeY);
	cpuopacity = (int *)malloc(sizeof(int) * frameSizeX * frameSizeY);
	for (int i = 0; i < frameSizeX; i++)
	{
		for (int j = 0; j < frameSizeY; j++)
		{
			cpuscene[i * frameSizeY + j] = 0;
			cpuopacity[i * frameSizeY + j] = -1;
		}
	}
	int *sceneopacity;
	cudaMalloc(&sceneopacity, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemcpy(sceneopacity, cpuopacity, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyHostToDevice);
	int *gpuscene;
	cudaMalloc(&gpuscene, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemcpy(gpuscene, cpuscene, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyHostToDevice);
	free(cpuscene);
	free(cpuopacity);
	int **verticestouched;
	cudaMalloc(&verticestouched, sizeof(int *) * V);
	bool *boolean_zeros;
	boolean_zeros = (bool *)malloc(sizeof(int) * V);
	for (int i = 0; i < V; i++)
	{
		boolean_zeros[i] = false;
	}
	int *intialize_to_zeros;
	intialize_to_zeros = (int *)malloc(sizeof(int) * V);
	for (int i = 0; i < V; i++)
	{
		intialize_to_zeros[i] = 0;
	}
	int blocks = (ceil(float(numTranslations)/float(1024)));
	bool *vis;
	vis = (bool *)malloc(sizeof(bool) * V);
	cudaMemcpy(vis, boolean_zeros, sizeof(bool) * V, cudaMemcpyHostToHost);
	int *size_of_touched;
	// cudaMalloc(&size_of_touched, sizeof(int) * V);
	size_of_touched=(int*)malloc(sizeof(int)*V);
	cudaMemcpy(size_of_touched, intialize_to_zeros, sizeof(int) * V, cudaMemcpyHostToHost);
	for (int i = 0; i < numTranslations; i++)
	{
		int vertex = translations[i][0];
		bool *temp_vertex_list;
		cudaMalloc(&temp_vertex_list, sizeof(bool) * V);
		boolean_zeros[vertex] = true;
		cudaMemcpy(temp_vertex_list, boolean_zeros, sizeof(bool) * V, cudaMemcpyHostToDevice);
		boolean_zeros[vertex] = false;
		bool *temp_level_list;
		cudaMalloc(&temp_level_list, sizeof(bool) * V);
		boolean_zeros[vertex] = true;
		cudaMemcpy(temp_level_list, boolean_zeros, sizeof(bool) * V, cudaMemcpyHostToDevice);
		boolean_zeros[vertex] = false;
		bool *rear, *gpurear;
		rear = (bool *)malloc(sizeof(bool));
		rear[0] = false;

		cudaMalloc(&gpurear, sizeof(bool));
		if (!vis[vertex])
		{
			vis[vertex] = true;
			do
			{
				cudaMemset(gpurear, false, sizeof(bool));
				bfs_kernal<<<blocks, 1024>>>(gpucsr, gpuoffset, temp_vertex_list, temp_level_list, gpurear, V);
				cudaDeviceSynchronize();
				cudaMemcpy(rear, gpurear, sizeof(bool), cudaMemcpyDeviceToHost);
			} while (rear[0] != false);
			cudaFree(gpurear);
			cudaFree(temp_level_list);
			std::vector<int> touched_vector;
			bool *cpuver;
			cpuver = (bool *)malloc(sizeof(bool) * V);
			cudaMemcpy(cpuver, temp_vertex_list, sizeof(bool) * V, cudaMemcpyDeviceToHost);
			for (int i = 0; i < V; i++)
			{
				if (cpuver[i] == true)
				{
					touched_vector.push_back(i);
				}
			}
			// for(int i=0;i<touched_vector.size();i++)
			// printf("%d ",touched_vector[i]);
			// printf("\n");
			int size=touched_vector.size();
			size_of_touched[vertex] = size;
			int *updated_vertex_list;
			cudaMalloc(&updated_vertex_list, sizeof(int) * size);
			cudaMemcpy(updated_vertex_list,touched_vector.data(),sizeof(int)*size,cudaMemcpyHostToDevice);
			cudaFree(temp_vertex_list);
			touched_vector.clear();
			cudaMemcpy(verticestouched + vertex, &updated_vertex_list, sizeof(int *), cudaMemcpyHostToDevice);
		}
	}
	cudaFree(gpuoffset);
	cudaFree(gpucsr);
	int *no_translations;
	cudaMalloc(&no_translations, sizeof(int));
	cudaMemcpy(no_translations, &numTranslations, sizeof(int), cudaMemcpyHostToDevice);
	int *gpu_size_of_touched;
	cudaMalloc(&gpu_size_of_touched, sizeof(int) * V);
	cudaMemcpy(gpu_size_of_touched, size_of_touched, sizeof(int) * V, cudaMemcpyHostToDevice);
	int blk = (ceil(float(numTranslations)/float(1024)));
	final_kernal<<<blk, 1024>>>(gpuglobalx, gpuglobaly, gpuframex, gpuframey, gputransforms, no_translations, V, verticestouched, gpu_size_of_touched);
	cudaDeviceSynchronize();
	cudaFree(verticestouched);
	cudaFree(gputransforms);
	cudaFree(no_translations);
	cudaMemcpy(hGlobalCoordinatesX, gpuglobalx, sizeof(int) * V, cudaMemcpyDeviceToHost);
	cudaMemcpy(hGlobalCoordinatesY, gpuglobaly, sizeof(int) * V, cudaMemcpyDeviceToHost);
	int *scenesizex, *scenesizey;
	cudaMalloc(&scenesizex, sizeof(int));
	cudaMalloc(&scenesizey, sizeof(int));
	cudaMemcpy(scenesizex, &frameSizeX, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(scenesizey, &frameSizeY, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(hGlobalCoordinatesX, gpuglobalx, sizeof(int) * V, cudaMemcpyDeviceToHost);
	cudaMemcpy(hGlobalCoordinatesY, gpuglobaly, sizeof(int) * V, cudaMemcpyDeviceToHost);
	for (int i = 0; i < V; i++)
	{
		int *current_mesh;
		cudaMalloc(&current_mesh, sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i]);
		cudaMemcpy(current_mesh, hMesh[i], sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i], cudaMemcpyHostToDevice);
		int blk = (ceil(float((hFrameSizeX[i] * hFrameSizeY[i]))/float(1024)));
		final_mesh<<<blk, 1024>>>(hFrameSizeX[i], hFrameSizeY[i], hGlobalCoordinatesX[i], hGlobalCoordinatesY[i], hOpacity[i], sceneopacity, current_mesh, gpuscene, frameSizeX, frameSizeY, V);
		cudaDeviceSynchronize();
		cudaFree(current_mesh);
	}

	cudaFree(gpuopacity);
	cudaFree(gpuframex);
	cudaFree(gpuframey);
	cudaFree(gpumesh);
	cudaFree(gpuglobalx);
	cudaFree(gpuglobaly);
	cudaMemcpy(hFinalPng, gpuscene, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);
	cudaFree(gpuscene);
	// Do not change anything below this comment.
	// Code ends here.

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::micro> timeTaken = end - start;

	printf("execution time : %f\n", timeTaken);
	// Write output matrix to file.
	const char *outputFileName = argv[2];
	writeFile(outputFileName, hFinalPng, frameSizeX, frameSizeY);
}
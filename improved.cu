#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

#define PI 3.142857

const int gridSize = 4;
const int mx = 256;
const int my = 256;

typedef struct {
    double x, y;
} vector2;

__device__ vector2 findNearestCorner(int x, int y) {

    double xOffset, yOffset;

    xOffset = x % gridSize;
    yOffset = y % gridSize;

    double bx, by;
    
    bx = xOffset <= gridSize/2;
    by = yOffset <= gridSize/2;

    vector2 v;

    if (bx) {
        if (by) {
            v.x = x - xOffset + gridSize;
            v.y = y - yOffset + gridSize;
        }
        else {
            v.x = x - xOffset + gridSize;
            v.y = y - yOffset;
        }
    }
    else {
        if (by) {
            v.x = x - xOffset;
            v.y = y - yOffset + gridSize;
        }
        else {
            v.x = x - xOffset;
            v.y = y - yOffset;
        }
    }
    return v;
}

__global__ void perlinNoise(int mx, int my, int mapx, int mapy, double *deviceVectors, double *out) {
    
    int threadId = blockIdx.x*blockDim.x+threadIdx.x;
    if (threadId > (mx*my)){
    }
    else{
    	int x = (threadId % my);
    	
    	int y = int(threadId / my); 
    	
    	vector2 corner, offSetVector, cornerVector;
	
    	corner = findNearestCorner(x, y);
	
    	offSetVector.x = corner.x - ((double)x);
    	offSetVector.y = corner.y - ((double)y);

    	double cornerVectorAngle, value;
    
    	int vectorIndex = (int)(corner.y / gridSize)*((corner.x / gridSize) + 2);
    
    	cornerVectorAngle = (deviceVectors)[vectorIndex];
    	//printf("wtf %i\n",threadIdx.x);
        
        
    	cornerVector.y = sin(cornerVectorAngle);
    	cornerVector.x = cos(cornerVectorAngle);
        
        
    	value = (offSetVector.x*cornerVector.x + offSetVector.y*cornerVector.y);
    	value = ((value + 1)/2);
    
    	//printf("%f %f %f %f %f %f %f %d %d %f %d\n", value, offSetVector.x, offSetVector.y, cornerVector.x, cornerVector.y, corner.x, corner.y, x, y, cornerVectorAngle, vectorIndex);
        
    	(out)[threadId] = value;
    	//printf("%f \n", value);
    }
}

int main() {
    
    srand(141414);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int mapx, mapy, nDot;
    
    mapx = ceil(((double)mx)/gridSize);
    mapy = ceil(((double)my)/gridSize);

    nDot = (mapx + 1)*(mapy + 1);

    printf("mx=%i\nmy=%i\nmapx=%i\nmapy=%i\nnDot=%i\n",mx,my,mapx,mapy,nDot);

    double* vectors = (double*) malloc(nDot * sizeof(double));
    double* output = (double*) malloc((mx * my) * sizeof(double));

    for (int i = 0; i < nDot; i++) {
        vectors[i] = rand() * 2 * PI / RAND_MAX;
        //printf("%i\t%f\n", i, vectors[i]);
    }
    
    double *deviceVectors = NULL;
    cudaMalloc((void **)&deviceVectors, (nDot * sizeof(double)));
    
    double *deviceOutput = NULL;
    cudaMalloc((void **)&deviceOutput, ((mx * my) * sizeof(double)));
    
    //cudaMemcpy(deviceVectors, vectors, (nDot * sizeof(double)), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = ceil((mx * my) / threadsPerBlock); 
    
    // start
    
    cudaEventRecord(start);
    
    const int numStreams = 8;
    cudaStream_t streams[numStreams];
    
    for (int i = 0; i < numStreams; i++){
    	cudaStreamCreate(&(streams[i]));
    }
    
    int chunkSize = ceil((mx*my) / numStreams);
    
    for (int stream = 0; stream < numStreams; stream++){
    const int lower = chunkSize * stream;
    const int upper = min(lower + chunkSize , mx*my);
    const int width = upper - lower;
    
    cudaMemcpyAsync(deviceVectors + lower, vectors + lower, sizeof(double)*width, cudaMemcpyHostToDevice, streams[stream]);
    
    perlinNoise<<<blocksPerGrid, threadsPerBlock, 0, streams[stream]>>>(mx, my, mapx, mapy, deviceVectors + lower, deviceOutput);
    
    cudaMemcpyAsync(output + lower, deviceOutput + lower, sizeof(double)*width, cudaMemcpyDeviceToHost, streams[stream]);
    }
    // end

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time: %f milliseconds\n", milliseconds);
    
    //perlinNoise<<<blocksPerGrid, threadsPerBlock>>>(mx, my, mapx, mapy, deviceVectors, deviceOutput);
    
    
    //cudaMemcpy(output, deviceOutput, ((mx * my) * sizeof(double)), cudaMemcpyDeviceToHost);
    
    
    FILE *fptr;

    fptr = fopen("output.txt","w");

    for (int i = 0; i < my; i++){
    	for (int j = 0; j < mx; j++){
    		//printf("%f ", output[j+(i*j)]);
    		fprintf(fptr, "%f ", output[j+(i*j)]);
    	}
    	//printf("\n");
    	fprintf(fptr, "%s", "\n");
    }
    
    fclose(fptr);
    

    return 0;
}

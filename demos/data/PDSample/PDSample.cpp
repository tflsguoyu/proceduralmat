#include <iostream>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "PDSampling.h"

int main(int argc, char *argv[])
{

	char *OUTPUT_PATH = (char *)"pdsample.out";
	float radius = 0.01f;
	unsigned seed = static_cast<unsigned>(time(0));

	for (int i=1; i<argc; i++) {
		char *arg = argv[i];	

		if (!strcmp(arg,"-o")) {
			if (i+1 < argc) {
				OUTPUT_PATH = argv[++i];
				// std::cout << OUTPUT_PATH << std::endl;
			}
			else 
				assert(0);
		}
		else if (!strcmp(arg,"-r")) {
			if (i+1 < argc) {
				radius = atof(argv[++i]);
				// std::cout << radius << std::endl;
			}
			else 
				assert(0);
		}
		else if (!strcmp(arg,"-s")) {
			if (i+1 < argc) {
				seed = static_cast<unsigned>(atoi(argv[++i]));
				// std::cout << seed << std::endl;
			}
			else 
				assert(0);
		}
		else 
			break;
	}

	if (radius<0.0005 || radius>.2) {
		printf("Radius (%f) is outside allowable range.\n", radius);
		exit(1);
	}

	PDSampler *sampler = new DartThrowing(seed, radius, true, 1000, 1);
	
	sampler->complete();
	
	int N = (int) sampler->points.size();
	FILE *output = fopen(OUTPUT_PATH,"wb");

	fwrite(&N, 4, 1, output);
	fwrite(&radius, 4, 1, output);
	for (int i=0; i<N; i++) {
		fwrite(&sampler->points[i], 8, 1, output);
	}
	fclose(output);

	delete sampler;

	return 0;
}

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/tuple.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

extern int ui_denoise_method;
extern int ui_filterSize;
extern int ui_iterations;
extern float ui_colorWeight;
extern float ui_normalWeight;
extern float ui_positionWeight;

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line)
{
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err)
	{
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file)
	{
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
	getchar();
#endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
	thrust::default_random_engine
	makeSeededRandomEngine(int iter, int index, int depth)
{
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4 *pbo, glm::ivec2 resolution,
							   int iter, glm::vec3 *image)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y)
	{
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

__global__ void gbufferToPBO(uchar4 *pbo, glm::ivec2 resolution, GBufferPixel *gBuffer, const int mod,
							 glm::vec3 minPos, glm::vec3 maxPos)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y)
	{
		int index = x + (y * resolution.x);

		if (mod == 0)
		{
			glm::vec3 scaledPos = (gBuffer[index].position - minPos) / (maxPos - minPos);
			scaledPos = glm::clamp(scaledPos, 0.0f, 1.0f); // Ensure values stay in [0, 1]

			pbo[index].x = scaledPos.x * 255;
			pbo[index].y = scaledPos.y * 255;
			pbo[index].z = scaledPos.z * 255;
			pbo[index].w = 0;
		}
		else if (mod == 1)
		{
			glm::vec3 scaledNormal = (gBuffer[index].normal + 1.0f) * 0.5f * 255.0f; // Map [-1, 1] to [0, 255]
			scaledNormal = glm::clamp(scaledNormal, 0.0f, 255.0f);

			pbo[index].x = scaledNormal.x;
			pbo[index].y = scaledNormal.y;
			pbo[index].z = scaledNormal.z;
			pbo[index].w = 0;
		}
	}
}

static void generateGaussianKernel(float *kernel, int kernelRadius, float sigma = 1.0f)
{
	int size = 2 * kernelRadius + 1;
	float sum = 0.0f;

	for (int y = -kernelRadius; y <= kernelRadius; ++y)
	{
		for (int x = -kernelRadius; x <= kernelRadius; ++x)
		{
			float exponent = -(x * x + y * y) / (2.0f * sigma * sigma);
			kernel[(y + kernelRadius) * size + (x + kernelRadius)] = exp(exponent);
			sum += kernel[(y + kernelRadius) * size + (x + kernelRadius)];
		}
	}

	for (int i = 0; i < size * size; ++i)
	{
		kernel[i] /= sum;
	}
}

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Material *dev_materials = NULL;
static PathSegment *dev_paths = NULL;
static ShadeableIntersection *dev_intersections = NULL;
static GBufferPixel *dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static glm::vec3 *dev_denoised_image_in = NULL;
static glm::vec3 *dev_denoised_image_out = NULL;

static float *dev_gaussian_kernal = NULL;
static int gaussian_kernel_radius = 0;

static float *dev_atrous_kernal = NULL;

void pathtraceInit(Scene *scene)
{
	hst_scene = scene;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

	// TODO: initialize any extra device memeory you need

	cudaMalloc(&dev_denoised_image_out, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoised_image_out, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_denoised_image_in, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_denoised_image_in, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_atrous_kernal, 25 * sizeof(float));
	cudaMemcpy(dev_atrous_kernal, atrous_kernel, 25 * sizeof(float), cudaMemcpyHostToDevice);

	// for Gaussian
	gaussian_kernel_radius = ui_filterSize / 2;
	const int kernelSize = 2 * gaussian_kernel_radius + 1;

	float *gaussian_host_kernal = new float[kernelSize * kernelSize];
	generateGaussianKernel(gaussian_host_kernal, gaussian_kernel_radius);

	cudaMalloc(&dev_gaussian_kernal, kernelSize * kernelSize * sizeof(float));
	cudaMemcpy(dev_gaussian_kernal, gaussian_host_kernal, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
	delete (gaussian_host_kernal);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
	cudaFree(dev_image); // no-op if dev_image is null
	cudaFree(dev_paths);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_gBuffer);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_denoised_image_out);
	cudaFree(dev_denoised_image_in);

	cudaFree(dev_gaussian_kernal);
	cudaFree(dev_atrous_kernal);

	checkCUDAError("pathtraceFree");
}

/**
 * Generate PathSegments with rays from the camera through the screen into the
 * scene, which is the first bounce of rays.
 *
 * Antialiasing - add rays for sub-pixel sampling
 * motion blur - jitter rays "in time"
 * lens effect - jitter ray origin positions based on a lens
 */
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment *pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y)
	{
		int index = x + (y * cam.resolution.x);
		PathSegment &segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		segment.ray.direction = glm::normalize(cam.view - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f) - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f));

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void computeIntersections(
	int depth, int num_paths, PathSegment *pathSegments, Geom *geoms, int geoms_size, ShadeableIntersection *intersections)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom &geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			// The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
		}
	}
}

__global__ void shadeSimpleMaterials(
	int iter, int num_paths, ShadeableIntersection *shadeableIntersections, PathSegment *pathSegments, Material *materials)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment segment = pathSegments[idx];
		if (segment.remainingBounces == 0)
		{
			return;
		}

		if (intersection.t > 0.0f)
		{ // if the intersection exists...
			segment.remainingBounces--;
			// Set up the RNG
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, segment.remainingBounces);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f)
			{
				segment.color *= (materialColor * material.emittance);
				segment.remainingBounces = 0;
			}
			else
			{
				segment.color *= materialColor;
				glm::vec3 intersectPos = intersection.t * segment.ray.direction + segment.ray.origin;
				scatterRay(segment, intersectPos, intersection.surfaceNormal, material, rng);
			}
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
		}
		else
		{
			segment.color = glm::vec3(0.0f);
			segment.remainingBounces = 0;
		}

		pathSegments[idx] = segment;
	}
}

__global__ void generateGBuffer(
	int num_paths,
	ShadeableIntersection *shadeableIntersections,
	PathSegment *pathSegments,
	GBufferPixel *gBuffer)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		gBuffer[idx].position =
			shadeableIntersections[idx].t * pathSegments[idx].ray.direction + pathSegments[idx].ray.origin;
		gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image, PathSegment *iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void gaussianBlur(glm::ivec2 resolution, glm::vec3 *inputImage, glm::vec3 *outputImage, int kernelRadius, float *kernel)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y)
	{
		int index = x + (y * resolution.x);

		glm::vec3 colorSum = glm::vec3(0.0f);
		float weightSum = 0.0f;

		// Apply the Gaussian blur filter
		for (int j = -kernelRadius; j <= kernelRadius; ++j)
		{
			for (int i = -kernelRadius; i <= kernelRadius; ++i)
			{
				int neighborX = glm::clamp(x + i, 0, resolution.x - 1);
				int neighborY = glm::clamp(y + j, 0, resolution.y - 1);
				int neighborIndex = neighborX + (neighborY * resolution.x);

				float weight = kernel[(j + kernelRadius) * (2 * kernelRadius + 1) + (i + kernelRadius)];
				colorSum += inputImage[neighborIndex] * weight;
				weightSum += weight;
			}
		}

		outputImage[index] = colorSum / weightSum;
	}
}

// An edge-avoiding a-trous denoising algorithm based on https://jo.dreggn.org/home/2010_atrous.pdf
__global__ void aTrousFilter(
	glm::ivec2 resolution,
	glm::vec3 *inImage,
	glm::vec3 *outImage,
	GBufferPixel *gBuffer,
	float *kernel,
	float c_phi, float n_phi, float p_phi,
	float stepwidth, bool avoidEdge)
{
	int idx_x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int idx_y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (idx_x >= resolution.x || idx_y >= resolution.y)
	{
		return;
	}

	glm::vec3 sum = glm::vec3(0.f); // accumulate color
	float sumWeight = 0.f;

	// read GBuffer values for the current pixel
	int idx = idx_y * resolution.x + idx_x;
	glm::vec3 cval = inImage[idx];
	glm::vec3 nval = gBuffer[idx].normal;
	glm::vec3 pval = gBuffer[idx].position;

	for (int i = -2; i <= 2; i++) // default to using a 5x5 kernel
	{
		for (int j = -2; j <= 2; j++)
		{
			// find neighboring pixel at desired offset (clamp to image boundaries)
			int xOffset = idx_x + j * stepwidth;
			int yOffset = idx_y + i * stepwidth;
			glm::ivec2 uv = glm::clamp(glm::ivec2(xOffset, yOffset), glm::ivec2(0), resolution);

			float weight = 1.f;
			float h = kernel[(i + 2) * 5 + (j + 2)];

			// color difference between current and neighboring pixel
			int currIdx = uv.x + uv.y * resolution.x;
			glm::vec3 ctemp = inImage[currIdx];
			glm::vec3 t = cval - ctemp;
			float dist2 = dot(t, t);
			float c_w = min(exp(-dist2 / c_phi), 1.f);

			if (avoidEdge)
			{
				// normal difference
				glm::vec3 ntemp = gBuffer[currIdx].normal;
				t = nval - ntemp;
				dist2 = max(dot(t, t) / (stepwidth * stepwidth), 0.f);
				float n_w = min(exp(-dist2 / n_phi), 1.f);

				// position difference
				glm::vec3 ptemp = gBuffer[currIdx].position;
				t = pval - ptemp;
				dist2 = dot(t, t);
				float p_w = min(exp(-dist2 / p_phi), 1.f);

				// calculate weights
				weight = c_w * n_w * p_w;
			}

			sum += ctemp * weight * h;
			sumWeight += weight * h;
		}
	}
	outImage[idx] = sum / sumWeight;
}

static void denoise(const Camera &cam)
{
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	if (ui_denoise_method == 1)
	{
		// Gaussian blur
		gaussianBlur<<<blocksPerGrid2d, blockSize2d>>>(cam.resolution, dev_image, dev_denoised_image_out, gaussian_kernel_radius, dev_gaussian_kernal);
	}
	else if (ui_denoise_method == 2 || ui_denoise_method == 3)
	{
		float c_phi = ui_colorWeight;
		float n_phi = ui_normalWeight;
		float p_phi = ui_positionWeight;
		float filterSize = ui_filterSize;
		bool avoidEdge = ui_denoise_method == 3;

		// calculate iterations based on filter size
		int numIter = filterSize < 5 ? 0 : floor(log2(filterSize / 5.f));

		// copy initial input image
		int pixelCount = cam.resolution.x * cam.resolution.y;
		cudaMemcpy(dev_denoised_image_in, dev_image, pixelCount * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

		for (int i = 0; i <= numIter; i++)
		{
			float stepwidth = 1 << i;
			aTrousFilter<<<blocksPerGrid2d, blockSize2d>>>(cam.resolution, dev_denoised_image_in, dev_denoised_image_out,
														   dev_gBuffer, dev_atrous_kernal, c_phi, n_phi, p_phi, stepwidth, avoidEdge);

			// ping pong buffer, don't swap at the last iteration
			if (i != numIter - 1)
			{
				std::swap(dev_denoised_image_in, dev_denoised_image_out);
			}
		}
	}
	cudaMemcpy(dev_image, dev_denoised_image_out,
			   cam.resolution.x * cam.resolution.y * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter)
{
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Pathtracing Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * NEW: For the first depth, generate geometry buffers (gbuffers)
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally:
	//     * if not denoising, add this iteration's results to the image
	//     * TODO: if denoising, run kernels that take both the raw pathtraced result and the gbuffer, and put the result in the "pbo" from opengl

	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment *dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	// Empty gbuffer
	cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	bool iterationComplete = false;
	while (!iterationComplete)
	{

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(
			depth, num_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_intersections);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();

		if (depth == 0)
		{
			generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_paths, dev_gBuffer);
		}

		depth++;

		shadeSimpleMaterials<<<numblocksPathSegmentTracing, blockSize1d>>>(
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials);
		iterationComplete = depth == traceDepth;
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

	if (ui_denoise_method != 0 && iter == ui_iterations)
	{
		denoise(cam);
	}

	///////////////////////////////////////////////////////////////////////////

	// CHECKITOUT: use dev_image as reference if you want to implement saving denoised images.
	// Otherwise, screenshots are also acceptable.
	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
			   pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}

// Custom functor to find the minimum of two glm::vec3 vectors
struct pos_min_op
{
	__host__ __device__
		glm::vec3
		operator()(const glm::vec3 &a, const glm::vec3 &b) const
	{
		return glm::vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
	}
};

// Custom functor to find the maximum of two glm::vec3 vectors
struct pos_max_op
{
	__host__ __device__
		glm::vec3
		operator()(const glm::vec3 &a, const glm::vec3 &b) const
	{
		return glm::vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
	}
};

// Functor to extract positions from GBufferPixel
struct extract_position
{
	__host__ __device__
		glm::vec3
		operator()(const GBufferPixel &g) const
	{
		return g.position;
	}
};

// Function to find min and max positions using thrust
void findMinMaxPositions(GBufferPixel *dev_gBuffer, int pixelCount, glm::vec3 &minPos, glm::vec3 &maxPos)
{
	// Create thrust device pointers
	thrust::device_ptr<GBufferPixel> gBufferPtr(dev_gBuffer);

	// Use transform_reduce to find min position
	minPos = thrust::transform_reduce(
		gBufferPtr,
		gBufferPtr + pixelCount,
		extract_position(),
		glm::vec3(FLT_MAX, FLT_MAX, FLT_MAX), // Initial value for min reduction
		pos_min_op());

	// Use transform_reduce to find max position
	maxPos = thrust::transform_reduce(
		gBufferPtr,
		gBufferPtr + pixelCount,
		extract_position(),
		glm::vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX), // Initial value for max reduction
		pos_max_op());
}

// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4 *pbo, const int mod)
{
	const Camera &cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	glm::vec3 minPos = {};
	glm::vec3 maxPos = {};
	if (mod == 0)
	{
		findMinMaxPositions(dev_gBuffer, cam.resolution.x * cam.resolution.y, minPos, maxPos);
	}

	// CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
	gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, dev_gBuffer, mod, minPos, maxPos);
}

void showImage(uchar4 *pbo, int iter)
{
	const Camera &cam = hst_scene->state.camera;
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// Send results to OpenGL buffer for rendering
	sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}

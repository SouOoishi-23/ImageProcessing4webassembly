#include <stdio.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <stdlib.h>
#include <emscripten.h>
#include <cmath>
#include <math.h>
#include <wasm_simd128.h>
#include <xmmintrin.h>
#include <assert.h>
#include <unistd.h>
using namespace std;

#define HV 1
#define VHILine 0
#define VHIImage 0

extern "C" void EMSCRIPTEN_KEEPALIVE initialized()
{
	cout << "initialized" << endl;
}
//以下サンプル
/*

extern "C" int EMSCRIPTEN_KEEPALIVE Add(int a, int b)
{
	return a + b;
}

*/


int result[2];

inline int border_s(const int val) { return (val >= 0) ? val : -val - 1; }
inline int border_sv(const int val) { return (val >= 0) ? val : -val - 4; }
inline int border_e(const int val, const int maxval) { return (val <= maxval) ? val : 2 * maxval - val + 1; }
inline int border_ev(const int val, const int maxval) { return (val <= maxval) ? val : 2 * maxval - val + 4; }
inline int get_simd_ceil(int val, int simdwidth)
{
	int v = (val % simdwidth == 0) ? val : (val / simdwidth + 1) * simdwidth;
	return v;
}

struct s_mallinfo {
	int arena;    /* non-mmapped space allocated from system */
	int ordblks;  /* number of free chunks */
	int smblks;   /* always 0 */
	int hblks;    /* always 0 */
	int hblkhd;   /* space in mmapped regions */
	int usmblks;  /* maximum total allocated space */
	int fsmblks;  /* always 0 */
	int uordblks; /* total allocated space */
	int fordblks; /* total free space */
	int keepcost; /* releasable (via malloc_trim) space */
};

extern "C" {
	extern s_mallinfo mallinfo();
}

unsigned int getTotalMemory()
{
	return EM_ASM_INT(return HEAP8.length);
}

unsigned int getFreeMemory()
{
	s_mallinfo i = mallinfo();
	unsigned int totalMemory = getTotalMemory();
	unsigned int dynamicTop = (unsigned int)sbrk(0);
	return totalMemory - dynamicTop + i.fordblks;
}

EMSCRIPTEN_KEEPALIVE
extern "C" void usingMemory()
{
	cout << "before allocation" << endl;
	cout << "Total memory: " << getTotalMemory() << "bytes" << endl;
	cout << "Free memory: " << getFreeMemory() << "bytes" << endl;
	cout << "Used: " << getTotalMemory() - getFreeMemory() << "bytes (" << (getTotalMemory() - getFreeMemory()) * 100.0 / getTotalMemory() << "%)" << endl;
	assert(getTotalMemory() == 16777216);
	//assert(getFreeMemory() >= 10000000);
}

EMSCRIPTEN_KEEPALIVE
extern "C" size_t* mm_malloc(size_t length, int size)
{
	//return (size_t*)malloc(length * size);
	return (size_t*)_mm_malloc(length * size, 32);
}

EMSCRIPTEN_KEEPALIVE
extern "C" void mm_free(size_t* p)
{
	_mm_free(p);
	//free(p);
}

EMSCRIPTEN_KEEPALIVE
extern "C" void GaussianFilter(float* src, int width, int height, int channels, float sigma)
{
	const size_t size = width * height * channels;
	const int r = (int)3 * sigma;
	const int ksize = 2 * r + 1;
	const float norm = -1.f / (2.f * sigma * sigma);
	float* src_mat = (float*)_mm_malloc(sizeof(float) * size, 32);
	src_mat = src;
	float* kernel = (float*)_mm_malloc(sizeof(float) * ksize, 32);
	uint8_t* dst = (uint8_t*)_mm_malloc(sizeof(uint8_t) * size, 32);
	float* dst_f = (float*)_mm_malloc(sizeof(float) * size, 32);
	uint8_t* ptr = &dst[0]; // 出力
	float* s = src;
	const int wstep = width * 4;
	const int wstep0 = 0 * wstep;
	const int wstep1 = 1 * wstep;
	const int wstep2 = 2 * wstep;
	const int wstep3 = 3 * wstep;
	const int wstep4 = 4 * wstep;
	double sum = 0.0;
	for (int j = -r, index = 0; j <= r; j++)
	{
		float v = exp((j * j) * norm);
		sum += (double)v;
		kernel[index++] = v;
	}
	for (int j = -r, index = 0; j <= r; j++)
	{
		kernel[index] = (float)(kernel[index] / sum);
		index++;
	}
#if HV

	const int r_ = get_simd_ceil(r, 4);
	const int R = 4*r_;

	// h filter
	float* d = &dst_f[0];
	for (int j = 0; j < height; j++)
	{

		{
			float* si = s;
			for (int i = 0; i < R; i += 4)
			{
				__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
				for (int k = 0; k < ksize; k++)
				{
					int idx = border_sv(i + 4 * k - R);
					__m128 ms = (idx >= 0) ? wasm_v128_load(si + idx) : wasm_v128_load(si);
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
				}
				d[i] = mv[0];
				d[i + 1] = mv[1];
				d[i + 2] = mv[2];
			}
		}
		for (int i = R; i < 4 * width - R; i += 4)
		{
			__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
			float* si = s + i - R;
			for (int k = 0; k < ksize; k++)
			{
				__m128 ms = wasm_v128_load(si); // b, g, r, alpha
				__m128 mg = wasm_f32x4_splat(kernel[k]);
				__m128 tmp = wasm_f32x4_mul(ms, mg);
				mv = wasm_f32x4_add(tmp, mv);
				si += 4;
			}
			wasm_v128_store(d + i, mv);
		}
		{
			float* si = s;
			for (int i = wstep - R; i < wstep; i += 4)
			{
				__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
				for (int k = 0; k < ksize; k++)
				{
					int idx = border_ev(i + 4 * k - R, wstep - 4);
					__m128 ms = (idx >= 0) ? wasm_v128_load(si + idx) : wasm_v128_load(si + wstep - 4);
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
				}
				d[i + 0] = mv[0];
				d[i + 1] = mv[1];
				d[i + 2] = mv[2];
			}
		}

		s += wstep;
		d += wstep;
	}

	// v filter
	// w:height, h: 1の配列 => 要素数heightの配列
	float* buffer_line_rows = (float*)_mm_malloc(sizeof(float) * height, 32);
	float* b = &buffer_line_rows[0];

	for (int i = 0; i < width * 4; i += 1)
	{
		for (int j = 0; j < height; j++) b[j] = dst_f[j * wstep + i];
		ptr = &dst[i];
		for (int j = 0; j < r_; j++)
		{
			float v = 0.f;
			float* si = &b[0];
			for (int k = 0; k < ksize; k++)
			{
				int idx = border_s(j + k - r);
				v += (idx >= 0) ? kernel[k] * si[idx] : kernel[k] * b[0];
			}
			*ptr = v;
			ptr += wstep;
		}
		for (int j = r_; j < height - r_; j += 4)
		{
			__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
			float* bi = b + j - r;
			for (int k = 0; k < ksize; k++)
			{
				__m128 ms = wasm_v128_load(bi);
				__m128 mg = wasm_f32x4_splat(kernel[k]);
				__m128 tmp = wasm_f32x4_mul(ms, mg);
				mv = wasm_f32x4_add(tmp, mv);
				bi++;
			}

			ptr[wstep0] = (uint8_t)mv[0];
			ptr[wstep1] = (uint8_t)mv[1];
			ptr[wstep2] = (uint8_t)mv[2];
			ptr[wstep3] = (uint8_t)mv[3];
			ptr += wstep4;
		}
		for (int j = height - r_; j < height; j++)
		{
			float v = 0.f;
			float* si = &b[0];
			for (int k = 0; k < ksize; k++)
			{
				int idx = border_e(j + k - r, height-1);
				v += (idx >= 0) ? kernel[k] * si[idx] : kernel[k] * b[0];
			}
			*ptr = v;
			ptr += wstep;
		}
		i++;
		for (int j = 0; j < height; j++) b[j] = dst_f[j * wstep + i];
		ptr = &dst[i];
		for (int j = 0; j < r_; j++)
		{
			float v = 0.f;
			float* si = &b[0];
			for (int k = 0; k < ksize; k++)
			{
				int idx = border_s(j + k - r);
				v += (idx >= 0) ? kernel[k] * si[idx] : kernel[k] * b[0];
			}
			*ptr = v;
			ptr += wstep;
		}
		for (int j = r_; j < height - r_; j += 4)
		{
			__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
			float* bi = b + j - r;
			for (int k = 0; k < ksize; k++)
			{
				__m128 ms = wasm_v128_load(bi);
				__m128 mg = wasm_f32x4_splat(kernel[k]);
				__m128 tmp = wasm_f32x4_mul(ms, mg);
				mv = wasm_f32x4_add(tmp, mv);
				bi++;
			}

			ptr[wstep0] = (uint8_t)mv[0];
			ptr[wstep1] = (uint8_t)mv[1];
			ptr[wstep2] = (uint8_t)mv[2];
			ptr[wstep3] = (uint8_t)mv[3];
			ptr += wstep4;
		}
		for (int j = height - r_; j < height; j++)
		{
			float v = 0.f;
			float* si = &b[0];
			for (int k = 0; k < ksize; k++)
			{
				int idx = border_e(j + k - r, height-1);
				v += (idx >= 0) ? kernel[k] * si[idx] : kernel[k] * b[0];
			}
			*ptr = v;
			ptr += wstep;
		}
		i++;
		for (int j = 0; j < height; j++) b[j] = dst_f[j * wstep + i];
		ptr = &dst[i];
		for (int j = 0; j < r_; j++)
		{
			float v = 0.f;
			float* si = &b[0];
			for (int k = 0; k < ksize; k++)
			{
				int idx = border_s(j + k - r);
				v += (idx >= 0) ? kernel[k] * si[idx] : kernel[k] * b[0];
			}
			*ptr = v;
			ptr += wstep;
		}
		for (int j = r_; j < height - r_; j += 4)
		{
			__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
			float* bi = b + j - r;
			for (int k = 0; k < ksize; k++)
			{
				__m128 ms = wasm_v128_load(bi);
				__m128 mg = wasm_f32x4_splat(kernel[k]);
				__m128 tmp = wasm_f32x4_mul(ms, mg);
				mv = wasm_f32x4_add(tmp, mv);
				bi++;
			}

			ptr[wstep0] = (uint8_t)mv[0];
			ptr[wstep1] = (uint8_t)mv[1];
			ptr[wstep2] = (uint8_t)mv[2];
			ptr[wstep3] = (uint8_t)mv[3];
			ptr += wstep4;
		}
		for (int j = height - r_; j < height; j++)
		{
			float v = 0.f;
			float* si = &b[0];
			for (int k = 0; k < ksize; k++)
			{
				int idx = border_e(j + k - r, height - 1);
				v += (idx >= 0) ? kernel[k] * si[idx] : kernel[k] * b[0];
			}
			*ptr = v;
			ptr += wstep;
		}
		i++;
		ptr = &dst[i];

		for (int j = 0; j < height; j += 4)
		{
			ptr[wstep0] = 255;
			ptr[wstep1] = 255;
			ptr[wstep2] = 255;
			ptr[wstep3] = 255;
			ptr += wstep4;
		}
	}

	_mm_free(kernel);
	_mm_free(buffer_line_rows);
	_mm_free(dst_f);
	_mm_free(src_mat);
#endif
#if VHILine
	//cout << "VHILine" << endl;

	const int simdwidth = width;
	float* mat = (float*)_mm_malloc(sizeof(float) * width * 3, 32);
	float* b0 = &mat[0];
	float* b1 = &mat[width];
	float* b2 = &mat[width * 2];

	const int R = 4 * r;
	for (int j = r; j < height-r; j++)
	{
		//vfilter
		for (int i = 0, index = 0; i < width * 4; i += 16)// 4(simd)*4(ch)
		{
			const int e = j + r;
			float* si = s + i;
			{
				// 0
				__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);

				for (int k = 0; k < e + 1; k++)
				{
					int idx = (j + k - r) * wstep;
					__m128 ms = wasm_f32x4_make(*(si+idx), *(si + idx + 4), *(si + idx + 8), *(si + idx + 12));
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
				}
				float* sii = si + wstep * (j + e + 1 - r);
				for (int k = e+1; k < ksize; k++)
				{
					__m128 ms = wasm_f32x4_make(*sii, *(sii + 4), *(sii + 8), *(sii + 12));
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
					sii += wstep;
				}
				wasm_v128_store(b0 + index, mv);
			}
			si++;
			{
				// 1
				__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);

				for (int k = 0; k < e + 1; k++)
				{
					int idx = (j + k - r) * wstep;
					__m128 ms = wasm_f32x4_make(*(si + idx), *(si + idx + 4), *(si + idx + 8), *(si + idx + 12));
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
				}
				float* sii = si + wstep * (j + e + 1 - r);
				for (int k = e + 1; k < ksize; k++)
				{
					__m128 ms = wasm_f32x4_make(*sii, *(sii + 4), *(sii + 8), *(sii + 12));
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
					sii += wstep;
				}
				wasm_v128_store(b1 + index, mv);
			}
			si++;
			{
				// 2
				__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);

				for (int k = 0; k < e + 1; k++)
				{
					int idx = (j + k - r) * wstep;
					__m128 ms = wasm_f32x4_make(*(si + idx), *(si + idx + 4), *(si + idx + 8), *(si + idx + 12));
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
				}
				float* sii = si + wstep * (j + e + 1 - r);
				for (int k = e + 1; k < ksize; k++)
				{
					__m128 ms = wasm_f32x4_make(*sii, *(sii + 4), *(sii + 8), *(sii + 12));
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
					sii += wstep;
				}
				wasm_v128_store(b2 + index, mv);
			}
			index++;
		}

		//hfilter
		{
			ptr = &dst[0] + j * wstep + R;
			for (size_t i = r; i < width - r; i += 1)
			{

				float* bi0 = b0 + i - r;
				float* bi1 = b1 + i - r;
				float* bi2 = b2 + i - r;

				__m128 mv0 = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
				for (int k = 0; k < ksize; k++)
				{
					__m128 ms = wasm_v128_load(bi0); // [0], [1], [2], [3]
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv0 = wasm_f32x4_add(tmp, mv0);
					bi0++;
				}
				//wasm_v128_store(d0 + i, mv);

				__m128 mv1 = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
				for (int k = 0; k < ksize; k++)
				{
					__m128 ms = wasm_v128_load(bi1); // [0], [1], [2], [3]
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv1 = wasm_f32x4_add(tmp, mv1);
					bi1++;
				}
				//wasm_v128_store(d1 + i, mv);

				__m128 mv2 = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
				for (int k = 0; k < ksize; k++)
				{
					__m128 ms = wasm_v128_load(bi2); // [0], [1], [2], [3]
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv2 = wasm_f32x4_add(tmp, mv2);
					bi2++;
				}
				//wasm_v128_store(d2 + i, mv);

				for (int c = 0; c < 4; c++)
				{
					*ptr = (uint8_t)mv0[i];
					ptr++;
					*ptr = (uint8_t)mv1[i];
					ptr++;
					*ptr = (uint8_t)mv2[i];
					ptr++;
					*ptr = (uint8_t)255;
					ptr++;
				}
			}
		}
	}
	_mm_free(mat);
#endif
#if VHIImage
	float* mat = (float*)_mm_malloc(sizeof(float) * width * 3, 32);
	const int R = 4 * r;


	for (int j = 0; j < height; j++)
	{
		float* b0 = &mat[0];
		float* b1 = &mat[width];
		float* b2 = &mat[width * 2];
		// v filter
		for (int i = 0, index = 0; i < wstep; i+=4)
		{
			__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
			const float* si = s + i;

			for (int k = 0; k < ksize; k++)
			{
				int idx = (j + k * 4 - R) * wstep;
				__m128 ms = wasm_v128_load(si+idx);
				__m128 mg = wasm_f32x4_splat(kernel[k]);
				__m128 tmp = wasm_f32x4_mul(ms, mg);
				mv = wasm_f32x4_add(tmp, mv);
			}

			b0[index] = mv[0];
			b1[index] = mv[1];
			b2[index] = mv[2];
			index++;
		}
		// h filter
		cout << "H filter" << endl;
		ptr = &dst[0] + j * wstep + R;
		for (int i = R, index = 0; i < wstep - R; i+=4)
		{
			{
				__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
				float* bi = b0 + index;
				for (int k = 0; k < ksize; k++)
				{
					__m128 ms = wasm_v128_load(bi);
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
					bi++;
				}
				*ptr = (uint8_t)mv[index];
				ptr++;
			}

			cout << "index:" << index << endl;
			cout << "wstep:" << wstep << endl;
			{
				__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
				float* bi = b1 + index;

				for (int k = 0; k < ksize; k++)
				{
					__m128 ms = wasm_v128_load(bi);
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
					bi++;
				}
				*ptr = (uint8_t)mv[index];
				ptr++;
			}

			cout << "index:" << index << endl;
			cout << "wstep:" << wstep << endl;
			{
				__m128 mv = wasm_f32x4_const(0.f, 0.f, 0.f, 0.f);
				float* bi = b2 + index;

				for (int k = 0; k < ksize; k++)
				{
					__m128 ms = wasm_v128_load(bi);
					__m128 mg = wasm_f32x4_splat(kernel[k]);
					__m128 tmp = wasm_f32x4_mul(ms, mg);
					mv = wasm_f32x4_add(tmp, mv);
					bi++;
				}
				*ptr = (uint8_t)mv[index];
				ptr++;
			}
			*ptr = 255;
			ptr++;
			index++;
			cout << "index:" << index << endl;
			cout << "wstep:" << wstep << endl;
		}
	}
	_mm_free(mat);


#endif
	usingMemory();
	result[0] = (int)&dst[0];
	result[1] = size * sizeof(uint8_t);
}

EMSCRIPTEN_KEEPALIVE
extern "C" void GaussianFilterwithoutSIMD(float* src, int width, int height, int channels, float sigma)
{
	size_t size = width * height * channels * sizeof(float);
	uint8_t dst_array[size / sizeof(float)];
	uint8_t* dst = &dst_array[0];
	int r = (int)3 * sigma;
	float* begin = src;
	float kernel[2 * r + 1][2 * r + 1];
	for (int j = -r; j <= r; j++)
	{
		for (int i = -r; i <= r; i++)
		{
			kernel[j + r][i + r] = expf(-(j * j + i * i) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
		}
	}
	int wstep = channels * width;
	for (int c = 0; c < channels-1; c++)
	{
		for (int j = r; j < height - r; j++)
		{
			float* src_p = src + c + j * wstep + channels * r;
			dst = &dst_array[0] + c + j * wstep + channels * r;
			for (int i = r; i < width - r; i++)
			{
				float sum = 0;
				for (int l = -r; l <= r; l++)
				{
					for (int k = -r; k <= r; k++)
					{
						sum += kernel[l + r][k + r] * (*(src_p + channels * k + l * wstep));
					}
				}
				*dst = (uint8_t)sum;
				src_p += channels;
				dst += channels;
			}
		}
	}

	
	dst = &dst_array[0] + 3;
	for (int j = 0; j < height * width; j++)
	{
		*dst = 255;
		dst += channels;
	}
	result[0] = (int)&dst_array[0];
	result[1] = size;
}

EMSCRIPTEN_KEEPALIVE
extern "C" void SeparableGaussianFilterwithoutSIMD(float* src, int width, int height, int channels, float sigma)
{
	size_t size = width * height * channels * sizeof(float);
	uint8_t dst_array[size / sizeof(float)];
	float* tmp_p;
	tmp_p = (float*)malloc(size);
	memcpy(tmp_p, src, size);
	uint8_t* dst = &dst_array[0];
	int r = (int)3 * sigma;
	float* begin = src;
	float* tmp_begin = &tmp_p[0];
	float kernel[2 * r + 1];

	for (int j = -r; j <= r; j++)
	{
		kernel[j + r] = expf(-((float)j * j) / (2 * sigma * sigma)) / powf(2 * M_PI * sigma * sigma, 0.5);
	}

	for (int c = 0; c < channels - 1; c++)
	{
		cout << "channels" << endl;
		src = begin + c + r * (width + 1) * channels;
		tmp_p = tmp_begin + c + r * (width + 1) * channels;
		for (int j = r; j < height - r; j++)
		{
			src = begin + c + (j * width + r) * channels;
			tmp_p = tmp_begin + c + (j * width + r) * channels;
			for (int i = r; i < width - r; i++)
			{
				float sum = 0;
				for (int k = -r; k <= r; k++)
				{
					sum += kernel[r + k] * (*(src + channels * k * width));
				}
				*tmp_p = sum;
				tmp_p += channels;
				src += channels;
			}
			tmp_p += 2 * r * channels;
			src += 2 * r * channels;
		}

		tmp_p = tmp_begin + c + r * (width + 1) * channels;
		dst = &dst_array[0] + c + r * (width + 1) * channels;
		for (int j = r; j < height - r; j++)
		{
			for (int i = r; i < width - r; i++)
			{
				float sum = 0;

				for (int k = -r; k <= r; k++)
				{
					sum += kernel[k + r] * (*(tmp_p + channels * k));
				}

				*dst = (uint8_t)sum;
				tmp_p += channels;
				dst += channels;
			}
			tmp_p += 2 * r * channels;
			dst += 2 * r * channels;
		}
	}

	dst = &dst_array[0] + 3;
	for (int j = 0; j < height * width; j++)
	{
		*dst = 255;
		dst += channels;
	}

	result[0] = (int)&dst_array[0];
	result[1] = size;
}

EMSCRIPTEN_KEEPALIVE
extern "C" int getResultPtr()
{
	return result[0];
}

EMSCRIPTEN_KEEPALIVE
extern "C" int getResultSize()
{
	return result[1];
}

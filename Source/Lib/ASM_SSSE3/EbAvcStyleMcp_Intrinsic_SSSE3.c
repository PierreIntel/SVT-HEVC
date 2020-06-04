/*
* Copyright(c) 2018 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include "EbAvcStyleMcp_SSSE3.h"

#include "EbDefinitions.h"

#include "emmintrin.h"
#include "tmmintrin.h"
#include "avx512bwintrin.h"
#include "avx512fintrin.h"


EB_EXTERN EB_ALIGN(16) const EB_S8 EbHevcAvcStyleLumaIFCoeff8_SSSE3[]= {
    -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25,
     9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,
    -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18,
    18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2, 18, -2,
    -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9, -1,  9,
    25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1, 25, -1
};

const __uint8_t EbHevcAvcStyleLumaIFCoeff8_VNNI[]= {
    -1, 25,  9, -1, -1, 25,  9, -1, -1, 25,  9, -1, -1, 25,  9, -1, 
    -1, 25,  9, -1, -1, 25,  9, -1, -1, 25,  9, -1, -1, 25,  9, -1,
    -2, 18, 18, -2, -2, 18, 18, -2, -2, 18, 18, -2, -2, 18, 18, -2, 
    -2, 18, 18, -2, -2, 18, 18, -2, -2, 18, 18, -2, -2, 18, 18, -2, 
    -1,  9, 25, -1, -1,  9, 25, -1, -1,  9, 25, -1, -1,  9, 25, -1, 
    -1,  9, 25, -1, -1,  9, 25, -1, -1,  9, 25, -1, -1,  9, 25, -1, 
};

void PictureCopyKernel_SSSE3(
	EB_BYTE                  src,
	EB_U32                   srcStride,
	EB_BYTE                  dst,
	EB_U32                   dstStride,
	EB_U32                   areaWidth,
	EB_U32                   areaHeight,
	EB_U32                   bytesPerSample);

void AvcStyleLumaInterpolationFilterHorizontal_SSSE3_INTRIN(
    EB_BYTE refPic,
    EB_U32 srcStride,
    EB_BYTE dst,
    EB_U32 dstStride,
    EB_U32 puWidth,
    EB_U32 puHeight,
    EB_BYTE tempBuf,
    EB_U32 fracPos)
{
    (void)tempBuf;
    __m128i IFOffset, IFCoeff_1_0, IFCoeff_3_2, sum_clip_U8;
    EB_U32 width_cnt, height_cnt;
    EB_U32 IFShift = 5;

    fracPos <<= 5;
    IFOffset = _mm_set1_epi16(0x0010);
    IFCoeff_1_0 = _mm_load_si128((__m128i *)(EbHevcAvcStyleLumaIFCoeff8_SSSE3 + fracPos - 32));
    IFCoeff_3_2 = _mm_load_si128((__m128i *)(EbHevcAvcStyleLumaIFCoeff8_SSSE3 + fracPos - 16));

    if (!(puWidth & 15)) { // 16x
        __m128i ref0, ref1, ref2, ref3, ref01_lo, ref01_hi, ref23_lo, ref23_hi, sum_lo, sum_hi;

        for (height_cnt = 0; height_cnt < puHeight; ++height_cnt){
            for (width_cnt = 0; width_cnt < puWidth; width_cnt += 16) {
                ref0 = _mm_loadu_si128((__m128i *)(refPic + width_cnt - 1));
                ref1 = _mm_loadu_si128((__m128i *)(refPic + width_cnt));
                ref2 = _mm_loadu_si128((__m128i *)(refPic + width_cnt + 1));
                ref3 = _mm_loadu_si128((__m128i *)(refPic + width_cnt + 2));

                ref01_lo = _mm_unpacklo_epi8(ref0, ref1);
                ref01_hi = _mm_unpackhi_epi8(ref0, ref1);
                ref23_lo = _mm_unpacklo_epi8(ref2, ref3);
                ref23_hi = _mm_unpackhi_epi8(ref2, ref3);

                sum_lo = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(_mm_maddubs_epi16(ref01_lo, IFCoeff_1_0), _mm_maddubs_epi16(ref23_lo, IFCoeff_3_2)), IFOffset), IFShift);
                sum_hi = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(_mm_maddubs_epi16(ref01_hi, IFCoeff_1_0), _mm_maddubs_epi16(ref23_hi, IFCoeff_3_2)), IFOffset), IFShift);
                sum_clip_U8 = _mm_packus_epi16(sum_lo, sum_hi);
                _mm_storeu_si128((__m128i *)(dst + width_cnt), sum_clip_U8);
            }
            refPic += srcStride;
            dst += dstStride;
        }
    }
    else { //8x
        __m128i  sum01, sum23, sum;

        for (height_cnt = 0; height_cnt < puHeight; ++height_cnt){
            for (width_cnt = 0; width_cnt < puWidth; width_cnt += 8) {
                sum01 = _mm_maddubs_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(refPic + width_cnt - 1)),
                                                            _mm_loadl_epi64((__m128i *)(refPic + width_cnt))), IFCoeff_1_0);

                sum23 = _mm_maddubs_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(refPic + width_cnt + 1)),
                                                            _mm_loadl_epi64((__m128i *)(refPic + width_cnt + 2))), IFCoeff_3_2);

                sum = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(sum01, sum23), IFOffset), IFShift);
                sum_clip_U8 = _mm_packus_epi16(sum, sum);

                _mm_storel_epi64((__m128i *)(dst + width_cnt), sum_clip_U8);
            }
            refPic += srcStride;
            dst += dstStride;
        }

    }
}

void AvcStyleLumaInterpolationFilterVertical_SSSE3_INTRIN(
    EB_BYTE refPic,
    EB_U32 srcStride,
    EB_BYTE dst,
    EB_U32 dstStride,
    EB_U32 puWidth,
    EB_U32 puHeight,
    EB_BYTE tempBuf,
    EB_U32 fracPos)
{
    (void)tempBuf;
    __m128i IFOffset, IFCoeff_1_0, IFCoeff_3_2, sum_clip_U8;
    EB_U32 width_cnt, height_cnt;
    EB_U32 IFShift = 5;
    EB_U32 srcStrideSkip = srcStride;
    EB_BYTE refPicTemp, dstTemp;

    fracPos <<= 5;
    refPic -= srcStride;
    IFOffset = _mm_set1_epi16(0x0010);
    __m256i IFOffset_256 = _mm256_broadcast_i32x4(IFOffset);//_mm256_set1_epi16(0x0010);
    __m512i IFOffset_512 = _mm512_broadcast_i32x4(IFOffset);

    IFCoeff_1_0 = _mm_load_si128((__m128i *)(EbHevcAvcStyleLumaIFCoeff8_SSSE3 + fracPos - 32));
    IFCoeff_3_2 = _mm_load_si128((__m128i *)(EbHevcAvcStyleLumaIFCoeff8_SSSE3 + fracPos - 16));
    
    //VNNI
    __m512i IFCoeff_1_0_vnni = _mm512_broadcast_i32x4(_mm_load_si128((__m128i *)(EbHevcAvcStyleLumaIFCoeff8_VNNI + fracPos - 32)));
    //AVX512
    __m512i IFCoeff_1_0_512 = _mm512_broadcast_i64x2(IFCoeff_1_0);
    __m512i IFCoeff_3_2_512 = _mm512_broadcast_i64x2(IFCoeff_3_2);
    //AVX2
    __m256i IFCoeff_1_0_256 = _mm256_broadcast_i64x2(IFCoeff_1_0);
    __m256i IFCoeff_3_2_256 = _mm256_broadcast_i64x2(IFCoeff_3_2);
    width_cnt = puWidth;
    
    __m512i zero = _mm512_set1_epi32(0);
    __m512i sum_lo_0, sum_lo_1, sum_hi_0, sum_hi_1, ref0_512, refs_512, ref2s_512, ref3s_512, ref0_1_lo, ref2_3_lo, ref0_1_hi, ref2_3_hi, sum_clip_U8_512;
    __m256i sum_clip_U8_256;

    for (height_cnt = 0; height_cnt < puHeight; ++height_cnt) {
        width_cnt = puWidth;
        refPicTemp = refPic;
        dstTemp = dst;
        while(width_cnt > 0){
            if (width_cnt & 64) { //64x
                //VNNI
                ref0_512 = _mm512_loadu_si512((__m512i *)(refPicTemp));
                refs_512 = _mm512_loadu_si512((__m512i *)(refPicTemp + srcStride));
                ref2s_512 = _mm512_loadu_si512((__m512i *)(refPicTemp + 2 * srcStride));
                ref3s_512 = _mm512_loadu_si512((__m512i *)(refPicTemp + 3 * srcStride));
                
                ref0_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0),ref0_512);
                refs_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0),refs_512);
                ref2s_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0),ref2s_512);
                ref3s_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0),ref3s_512);

                ref0_1_lo = _mm512_unpacklo_epi8(ref0_512, refs_512);
                ref2_3_lo = _mm512_unpacklo_epi8(ref2s_512, ref3s_512);
                ref0_1_hi = _mm512_unpackhi_epi8(ref0_512, refs_512);
                ref2_3_hi = _mm512_unpackhi_epi8(ref2s_512, ref3s_512);

                sum_lo_0 =  _mm512_dpbusd_epi32(zero, _mm512_unpacklo_epi16(ref0_1_lo, ref2_3_lo), IFCoeff_1_0_vnni); // 64[0]
                sum_lo_1 =  _mm512_dpbusd_epi32(zero, _mm512_unpackhi_epi16(ref0_1_lo, ref2_3_lo), IFCoeff_1_0_vnni); // 64[1]
                sum_hi_0 =  _mm512_dpbusd_epi32(zero, _mm512_unpacklo_epi16(ref0_1_hi, ref2_3_hi), IFCoeff_1_0_vnni); // 64[2]
                sum_hi_1 =  _mm512_dpbusd_epi32(zero, _mm512_unpackhi_epi16(ref0_1_hi, ref2_3_hi), IFCoeff_1_0_vnni); // 64[3]

                sum_lo_0 = _mm512_castsi256_si512(_mm512_cvtusepi32_epi16(sum_lo_0));
                sum_lo_0 = _mm512_inserti64x4(sum_lo_0, _mm512_cvtusepi32_epi16(sum_lo_1), 1);
                sum_hi_0 = _mm512_castsi256_si512(_mm512_cvtusepi32_epi16(sum_hi_0));
                sum_hi_0 = _mm512_inserti64x4(sum_hi_0, _mm512_cvtusepi32_epi16(sum_hi_1), 1);
                
                sum_lo_0 = _mm512_srai_epi16(_mm512_add_epi16(sum_lo_0, IFOffset_512), IFShift);
                sum_hi_0 = _mm512_srai_epi16(_mm512_add_epi16(sum_hi_0, IFOffset_512), IFShift);
                
                sum_clip_U8_512 = _mm512_castsi256_si512(_mm512_cvtusepi16_epi8(sum_lo_0));
                sum_clip_U8_512 = _mm512_inserti64x4(sum_clip_U8_512, _mm512_cvtusepi16_epi8(sum_hi_0), 1);
                __m512i vnni = sum_clip_U8_512;

                //AVX512
                ref0_512 = _mm512_loadu_si512((__m512i *)(refPicTemp));
                refs_512 = _mm512_loadu_si512((__m512i *)(refPicTemp + srcStride));
                ref2s_512 = _mm512_loadu_si512((__m512i *)(refPicTemp + 2 * srcStride));
                ref3s_512 = _mm512_loadu_si512((__m512i *)(refPicTemp + 3 * srcStride));

                ref0_1_lo = _mm512_unpacklo_epi8(ref0_512, refs_512);
                ref2_3_lo = _mm512_unpacklo_epi8(ref2s_512, ref3s_512);
                ref0_1_hi = _mm512_unpackhi_epi8(ref0_512, refs_512);
                ref2_3_hi = _mm512_unpackhi_epi8(ref2s_512, ref3s_512);

                sum_lo_0 = _mm512_add_epi16(_mm512_maddubs_epi16(ref0_1_lo,IFCoeff_1_0_512),
                                        _mm512_maddubs_epi16(ref2_3_lo, IFCoeff_3_2_512));

                sum_hi_0 = _mm512_add_epi16(_mm512_maddubs_epi16(ref0_1_hi,IFCoeff_1_0_512),
                                        _mm512_maddubs_epi16(ref2_3_hi, IFCoeff_3_2_512));

                sum_lo_0 = _mm512_srai_epi16(_mm512_add_epi16(sum_lo_0, IFOffset_512), IFShift);
                sum_hi_0 = _mm512_srai_epi16(_mm512_add_epi16(sum_hi_0, IFOffset_512), IFShift);
                sum_clip_U8_512 = _mm512_packus_epi16(sum_lo_0, sum_hi_0);
                
                //COMPAR
                if(_mm512_cmpneq_epi8_mask(sum_clip_U8_512, vnni)){
                    printf("Error 64\n");
                    printf("vnni = %llx %llx %llx %llx %llx %llx %llx %llx\n", vnni[7], vnni[6],vnni[5], vnni[4],vnni[3], vnni[2],vnni[1], vnni[0]);
                    printf("sum  = %llx %llx %llx %llx %llx %llx %llx %llx\n", sum_clip_U8_512[7], sum_clip_U8_512[6],sum_clip_U8_512[5], sum_clip_U8_512[4],sum_clip_U8_512[3], sum_clip_U8_512[2],sum_clip_U8_512[1], sum_clip_U8_512[0]);
                }
                
                //STORE RESULT
                _mm512_storeu_si512((__m512i *)(dstTemp), sum_clip_U8_512);

                refPicTemp += 64;
                dstTemp += 64;
                width_cnt -= 64;
            }
            if (width_cnt & 32) {
                
                //VNNI
                ref0_512 = _mm512_castsi256_si512(_mm256_loadu_si256((__m256i *)(refPicTemp)));
                refs_512 = _mm512_castsi256_si512(_mm256_loadu_si256((__m256i *)(refPicTemp + srcStride)));
                ref0_512 = _mm512_inserti64x4(ref0_512,_mm256_loadu_si256((__m256i *)(refPicTemp + 2 * srcStride)),1);
                refs_512 = _mm512_inserti64x4(ref0_512,_mm256_loadu_si256((__m256i *)(refPicTemp + 3 * srcStride)),1);

                ref0_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0),ref0_512);
                refs_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0),refs_512);

                ref0_1_lo = _mm512_unpacklo_epi8(ref0_512, refs_512);
                ref2_3_hi = _mm512_unpackhi_epi8(ref0_512, refs_512);

                sum_lo_0 =  _mm512_dpbusd_epi32(zero, _mm512_unpacklo_epi16(ref0_1_lo, ref2_3_hi), IFCoeff_1_0_vnni);
                sum_hi_0 =  _mm512_dpbusd_epi32(zero, _mm512_unpacklo_epi16(ref0_1_lo, ref2_3_hi), IFCoeff_1_0_vnni);

                sum_lo_0 = _mm512_castsi256_si512(_mm512_cvtusepi32_epi16(sum_lo_0));
                sum_lo_0 = _mm512_inserti64x4(sum_lo_0, _mm512_cvtusepi32_epi16(sum_hi_0), 1);
                sum_lo_0 = _mm512_srai_epi16(_mm512_add_epi16(sum_lo_0, IFOffset_512), IFShift);
                
                sum_clip_U8_256 = _mm512_cvtusepi16_epi8(sum_lo_0);
                __m256i vnni = sum_clip_U8_256;
                
                //AVX2
                __m256i ref0_256 = _mm256_loadu_si256((__m512i *)(refPicTemp));
                __m256i refs_256 = _mm256_loadu_si256((__m512i *)(refPicTemp + srcStride));
                __m256i ref2s_256 = _mm256_loadu_si256((__m512i *)(refPicTemp + 2 * srcStride));
                __m256i ref3s_256 = _mm256_loadu_si256((__m512i *)(refPicTemp + 3 * srcStride));

                __m256i ref0_1_lo = _mm256_unpacklo_epi8(ref0_256, refs_256);
                __m256i ref2_3_lo = _mm256_unpacklo_epi8(ref2s_256, ref3s_256);
                __m256i ref0_1_hi = _mm256_unpackhi_epi8(ref0_256, refs_256);
                __m256i ref2_3_hi = _mm256_unpackhi_epi8(ref2s_256, ref3s_256);

                __m256i sum_lo_0 = _mm256_add_epi16(_mm256_maddubs_epi16(ref0_1_lo,IFCoeff_1_0_256),
                                        _mm256_maddubs_epi16(ref2_3_lo, IFCoeff_3_2_256));

                __m256i sum_hi_0 = _mm256_add_epi16(_mm256_maddubs_epi16(ref0_1_hi,IFCoeff_1_0_256),
                                        _mm256_maddubs_epi16(ref2_3_hi, IFCoeff_3_2_256));

                sum_lo_0 = _mm256_srai_epi16(_mm256_add_epi16(sum_lo_0, IFOffset_256), IFShift);
                sum_hi_0 = _mm256_srai_epi16(_mm256_add_epi16(sum_hi_0, IFOffset_256), IFShift);
                sum_clip_U8_256 = _mm256_packus_epi16(sum_lo_0, sum_hi_0);

                //COMPAR
                if(_mm256_cmpneq_epi8_mask(sum_clip_U8_256, vnni)){
                    printf("Error 32\n");
                }

                //STORE RESULT
                _mm256_storeu_si256((__m256i *)(dstTemp), sum_clip_U8_256);
                refPicTemp += 32;
                dstTemp += 32;
                width_cnt -= 32;
            }    
            if (width_cnt & 16) { //16x
                //VNNI  
                /*
                ref0_512 = _mm512_castsi128_si512(_mm_loadu_si128((__m128i *)(refPicTemp)));
                refs_512 = _mm512_castsi128_si512(_mm_loadu_si128((__m128i *)(refPicTemp) + srcStride));
                ref0_512 = _mm512_inserti64x2(ref0_512,_mm_loadu_si128((__m128i *)(refPicTemp + 2 * srcStride)),2);
                refs_512 = _mm512_inserti64x2(refs_512,_mm_loadu_si128((__m128i *)(refPicTemp + 3 * srcStride)),2);

                //ref0_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0),ref0_512);
                //refs_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(15,11,7,3,14,10,6,2,13,9,5,1,12,8,4,0),refs_512);
                
                ref0_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(11,11,3,3,10,10,2,2,9,9,5,1,8,8,0,0),ref0_512);
                refs_512 = _mm512_permutexvar_epi32(_mm512_set_epi32(11,11,3,3,10,10,2,2,9,9,5,1,8,8,0,0),refs_512);

                ref0_1_lo = _mm512_unpacklo_epi8(ref0_512, refs_512);
                ref2_3_hi = _mm512_unpackhi_epi8(ref0_512, refs_512);

                sum_lo_0 =  _mm512_dpbusd_epi32(zero, _mm512_unpacklo_epi16(ref0_1_lo, ref2_3_hi), IFCoeff_1_0_vnni);
                sum_clip_U8_256 = _mm512_cvtusepi32_epi16(sum_lo_0);
                sum_clip_U8_256 = _mm256_srai_epi16(_mm256_add_epi16(sum_clip_U8_256, IFOffset_256), IFShift);
                sum_clip_U8 = _mm256_cvtusepi16_epi8(sum_clip_U8_256);
                __m128i vnni = sum_clip_U8;
                */
                //SSSE3
                __m128i sum_lo, sum_hi, ref0, refs, ref2s, ref3s;   
                ref0 = _mm_loadu_si128((__m128i *)(refPicTemp));
                refs = _mm_loadu_si128((__m128i *)(refPicTemp + srcStride));
                ref2s = _mm_loadu_si128((__m128i *)(refPicTemp + 2 * srcStride));
                ref3s = _mm_loadu_si128((__m128i *)(refPicTemp + 3 * srcStride));

                sum_lo = _mm_add_epi16(_mm_maddubs_epi16(_mm_unpacklo_epi8(ref0, refs), IFCoeff_1_0),
                    _mm_maddubs_epi16(_mm_unpacklo_epi8(ref2s, ref3s), IFCoeff_3_2));

                sum_hi = _mm_add_epi16(_mm_maddubs_epi16(_mm_unpackhi_epi8(ref0, refs), IFCoeff_1_0),
                    _mm_maddubs_epi16(_mm_unpackhi_epi8(ref2s, ref3s), IFCoeff_3_2));

                sum_lo = _mm_srai_epi16(_mm_add_epi16(sum_lo, IFOffset), IFShift);
                sum_hi = _mm_srai_epi16(_mm_add_epi16(sum_hi, IFOffset), IFShift);
                sum_clip_U8 = _mm_packus_epi16(sum_lo, sum_hi);

                //COMPAR
                /*if(_mm_cmpeq_epi8_mask(sum_clip_U8, vnni)){
                    printf("Error 16\n");
                    printf("vnni = %llx %llx\n", vnni[1], vnni[0]);
                    printf("sse3 = %llx %llx\n", sum_clip_U8[1], sum_clip_U8[0]);
                }*/

                //STORE RESULT
                _mm_storeu_si128((__m128i *)(dstTemp), sum_clip_U8);

                refPicTemp += 16;
                dstTemp += 16;
                width_cnt -= 16;    
            }
            if (width_cnt & 8){
                 __m128i sum, sum01, sum23;
            
                sum01 = _mm_maddubs_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(refPicTemp)),
                                                            _mm_loadl_epi64((__m128i *)(refPicTemp + srcStride))), IFCoeff_1_0);

                sum23 = _mm_maddubs_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((__m128i *)(refPicTemp + 2 * srcStride)),
                                                            _mm_loadl_epi64((__m128i *)(refPicTemp + 3 * srcStride))), IFCoeff_3_2);

                sum = _mm_srai_epi16(_mm_add_epi16(_mm_add_epi16(sum01, sum23), IFOffset), IFShift);
                sum_clip_U8 = _mm_packus_epi16(sum, sum);
                _mm_storel_epi64((__m128i *)(dstTemp), sum_clip_U8);
                refPicTemp += 8;
                dstTemp += 8;
                width_cnt -= 8; 
            }
        }
        dst += dstStride;
        refPic += srcStrideSkip;
    }
}

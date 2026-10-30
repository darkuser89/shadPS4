// SPDX-License-Identifier: MIT
/*
    Copyright (c) Microsoft Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/
// Copyright 2024 Valve Corporation
// Copyright 2026 shadPS4 Emulator Project
// Metal adaptation of the Microsoft D3D11 fixed-function tessellator (Mesa poly/cl).

#pragma once

#include <string_view>

namespace Metal {

inline constexpr std::string_view TessellatorSource = R"metal(
#include <metal_stdlib>
using namespace metal;
constant uint MaxPoints = 4225;
constant uint MaxIndices = 24576;
enum Partitioning {
    FractionalOdd,
    FractionalEven,
    Integer,
};

struct DomainPoint {
    uint u;
    uint v;
};
struct PatchFactors {
    uint outer[4];
    uint inner[2];
    uint instance;
    uint patch;
};
struct DomainPatch {
    uint point_count;
    uint index_count;
    uint instance;
    uint patch;
};
struct TessStats {
    DomainPatch metadata;
    uint written_points;
    uint written_indices;
    uint max_index;
    uint errors;
};
struct TessParams {
    float factors[6];
    device DomainPoint* points;
    device uint* indices;
    thread TessStats* stats;
    Partitioning partitioning;
    bool points_mode;
    bool ccw;
    bool triangle;
};
struct Parameters {
    uint type;
    uint partitioning;
    uint topology;
    uint patches;
    float minimum_factor;
    float maximum_factor;
};
static_assert(sizeof(DomainPoint) == 8);
static_assert(sizeof(PatchFactors) == 32);
static_assert(sizeof(DomainPatch) == 16);
static_assert(sizeof(Parameters) == 24);
#define TESS_MIN_ISOLINE_DENSITY_TESSELLATION_FACTOR 1.0f
#define TESS_MAX_ISOLINE_DENSITY_TESSELLATION_FACTOR 64.0f

typedef unsigned int FXP;

enum {
    U = 0,
    V = 1,
};

enum {
    Ueq0 = 0,
    Veq0 = 1,
    Weq0 = 2,
};

enum {
    Ueq1 = 2,
    Veq1 = 3,
};

#define QUAD_AXES 2
#define QUAD_EDGES 4
#define TRI_EDGES 3

typedef enum DIAGONALS {
    DIAGONALS_INSIDE_TO_OUTSIDE,
    DIAGONALS_INSIDE_TO_OUTSIDE_EXCEPT_MIDDLE,
    DIAGONALS_MIRRORED
} DIAGONALS;

typedef struct TESS_FACTOR_CONTEXT {
    FXP fxpInvNumSegmentsOnFloorTessFactor;
    FXP fxpInvNumSegmentsOnCeilTessFactor;
    FXP fxpHalfTessFactorFraction;
    int numHalfTessFactorPoints;
    int splitPointOnFloorHalfTessFactor;
} TESS_FACTOR_CONTEXT;

struct INDEX_PATCH_CONTEXT {
    int insidePointIndexDeltaToRealValue;
    int insidePointIndexBadValue;
    int insidePointIndexReplacementValue;
    int outsidePointIndexPatchBase;
    int outsidePointIndexDeltaToRealValue;
    int outsidePointIndexBadValue;
    int outsidePointIndexReplacementValue;
};

struct INDEX_PATCH_CONTEXT2 {
    int baseIndexToInvert;
    int indexInversionEndPoint;
    int cornerCaseBadValue;
    int cornerCaseReplacementValue;
};

struct CHWTessellator {
    thread const TessParams* params;

    device uint* Index;

    bool bUsingPatchedIndices;
    bool bUsingPatchedIndices2;
    struct INDEX_PATCH_CONTEXT IndexPatchCtx;
    struct INDEX_PATCH_CONTEXT2 IndexPatchCtx2;
};

#define FXP_INTEGER_BITS 15
#define FXP_FRACTION_BITS 16
#define FXP_FRACTION_MASK 0x0000ffff
#define FXP_INTEGER_MASK 0x7fff0000
#define FXP_ONE (1 << FXP_FRACTION_BITS)
#define FXP_ONE_THIRD 0x00005555
#define FXP_TWO_THIRDS 0x0000aaaa
#define FXP_ONE_HALF 0x00008000

static inline device uint* AllocateIndices(thread const TessParams* params, uint count) {
    if (count > MaxIndices) {
        params->stats->errors |= 1u;
        count = 0;
    }
    params->stats->metadata.index_count = count;
    return params->indices;
}
static inline void DrawPoints(thread const TessParams* params, uint count) {
    device uint* indices = AllocateIndices(params, count);
    for (uint i = 0; i < params->stats->metadata.index_count; ++i) {
        indices[i] = i;
        params->stats->max_index = i;
        ++params->stats->written_indices;
    }
}
static inline void DrawEmpty(thread const TessParams* params) {
    params->stats->metadata.point_count = 0;
    params->stats->metadata.index_count = 0;
}
static inline void AllocatePoints(thread const TessParams* params, uint count) {
    if (count > MaxPoints) {
        params->stats->errors |= 1u;
        count = 0;
    }
    params->stats->metadata.point_count = count;
}

#define TESS_MIN_ODD_TESSELLATION_FACTOR 1
#define TESS_MAX_ODD_TESSELLATION_FACTOR 63
#define TESS_MIN_EVEN_TESSELLATION_FACTOR 2
#define TESS_MAX_EVEN_TESSELLATION_FACTOR 64

#define EPSILON 0.0000152587890625f
#define MIN_ODD_TESSFACTOR_PLUS_HALF_EPSILON (TESS_MIN_ODD_TESSELLATION_FACTOR + EPSILON / 2)

static inline float clamp_factor(float factor, Partitioning partitioning, float maxf) {
    float lower = (partitioning == FractionalEven) ? TESS_MIN_EVEN_TESSELLATION_FACTOR
                                                   : TESS_MIN_ODD_TESSELLATION_FACTOR;

    float upper = (partitioning == FractionalOdd) ? TESS_MAX_ODD_TESSELLATION_FACTOR
                                                  : TESS_MAX_EVEN_TESSELLATION_FACTOR;

    if (partitioning == FractionalOdd && maxf > MIN_ODD_TESSFACTOR_PLUS_HALF_EPSILON) {

        lower = TESS_MIN_ODD_TESSELLATION_FACTOR + EPSILON;
    }

    factor = fmin(fmax(factor, lower), upper);

    if (partitioning == Integer) {
        factor = ceil(factor);
    }

    return factor;
}

static inline FXP floatToFixed(const float input) {
    return uint(fma(input, float(FXP_ONE), 0.5f));
}

static inline bool isOdd(const float input) {
    return ((int)input) & 1;
}

static inline FXP fxpCeil(const FXP input) {
    if (input & FXP_FRACTION_MASK) {
        return (input & FXP_INTEGER_MASK) + FXP_ONE;
    }
    return input;
}

static inline FXP fxpFloor(const FXP input) {
    return (input & FXP_INTEGER_MASK);
}

static inline int PatchIndexValue(thread struct CHWTessellator* ctx, int index) {
    if (ctx->bUsingPatchedIndices) {

        if (index >= ctx->IndexPatchCtx.outsidePointIndexPatchBase) {
            if (index == ctx->IndexPatchCtx.outsidePointIndexBadValue)
                return ctx->IndexPatchCtx.outsidePointIndexReplacementValue;
            else
                return index + ctx->IndexPatchCtx.outsidePointIndexDeltaToRealValue;
        } else {
            if (index == ctx->IndexPatchCtx.insidePointIndexBadValue)
                return ctx->IndexPatchCtx.insidePointIndexReplacementValue;
            else
                return index + ctx->IndexPatchCtx.insidePointIndexDeltaToRealValue;
        }
    } else if (ctx->bUsingPatchedIndices2) {
        if (index == ctx->IndexPatchCtx2.cornerCaseBadValue) {
            return ctx->IndexPatchCtx2.cornerCaseReplacementValue;
        } else if (index >= ctx->IndexPatchCtx2.baseIndexToInvert) {
            return ctx->IndexPatchCtx2.indexInversionEndPoint - index;
        }
    }

    return index;
}

static inline void DefinePoint(thread const TessParams* params, int index, FXP u, FXP v) {
    if (uint(index) >= params->stats->metadata.point_count || u > FXP_ONE || v > FXP_ONE ||
        (params->triangle && u + v > FXP_ONE)) {
        params->stats->errors |= 2u;
        return;
    }
    params->points[index] = {u, v};
    ++params->stats->written_points;
}
static inline void DefineIndex(thread CHWTessellator* ctx, int index, int offset) {
    index = PatchIndexValue(ctx, index);
    thread TessStats* stats = ctx->params->stats;
    if (uint(offset) >= stats->metadata.index_count || uint(index) >= MaxPoints) {
        stats->errors |= 4u;
        return;
    }
    ctx->Index[offset] = uint(index);
    stats->max_index = max(stats->max_index, uint(index));
    ++stats->written_indices;
}
static inline void DefineTriangle(thread CHWTessellator* ctx, int i0, int i1, int i2, int offset) {
    DefineIndex(ctx, i0, offset);
    DefineIndex(ctx, ctx->params->ccw ? i2 : i1, offset + 1);
    DefineIndex(ctx, ctx->params->ccw ? i1 : i2, offset + 2);
}

static inline uint RemoveMSB(uint val) {
    uint bit = val ? (1 << (31 - clz(val))) : 0;
    return val & ~bit;
}

static inline int NumPointsForTessFactor(bool odd, FXP fxpTessFactor) {

    FXP f = fxpTessFactor + (odd ? (FXP_ONE + 1) : 1);
    int r = fxpCeil(f / 2) >> (FXP_FRACTION_BITS - 1);
    return odd ? r : r + 1;
}

static inline void ComputeTessFactorCtx(bool odd, FXP fxpTessFactor,
                                        thread TESS_FACTOR_CONTEXT* TessFactorCtx) {

    FXP fxpHalfTessFactor = (fxpTessFactor + 1) / 2;
    if (odd || (fxpHalfTessFactor == FXP_ONE_HALF)) {
        fxpHalfTessFactor += FXP_ONE_HALF;
    }
    FXP fxpFloorHalfTessFactor = fxpFloor(fxpHalfTessFactor);
    FXP fxpCeilHalfTessFactor = fxpCeil(fxpHalfTessFactor);
    TessFactorCtx->fxpHalfTessFactorFraction = fxpHalfTessFactor - fxpFloorHalfTessFactor;
    TessFactorCtx->numHalfTessFactorPoints = (fxpCeilHalfTessFactor >> FXP_FRACTION_BITS);

    if (fxpCeilHalfTessFactor == fxpFloorHalfTessFactor) {
        TessFactorCtx->splitPointOnFloorHalfTessFactor = TessFactorCtx->numHalfTessFactorPoints + 1;
    } else if (odd) {
        if (fxpFloorHalfTessFactor == FXP_ONE) {
            TessFactorCtx->splitPointOnFloorHalfTessFactor = 0;
        } else {
            TessFactorCtx->splitPointOnFloorHalfTessFactor =
                (RemoveMSB((fxpFloorHalfTessFactor >> FXP_FRACTION_BITS) - 1) << 1) + 1;
        }
    } else {
        TessFactorCtx->splitPointOnFloorHalfTessFactor =
            (RemoveMSB(fxpFloorHalfTessFactor >> FXP_FRACTION_BITS) << 1) + 1;
    }
    int numFloorSegments = (fxpFloorHalfTessFactor * 2) >> FXP_FRACTION_BITS;
    int numCeilSegments = (fxpCeilHalfTessFactor * 2) >> FXP_FRACTION_BITS;
    if (odd) {
        numFloorSegments -= 1;
        numCeilSegments -= 1;
    }
    TessFactorCtx->fxpInvNumSegmentsOnFloorTessFactor =
        floatToFixed(1.0f / (float)numFloorSegments);
    TessFactorCtx->fxpInvNumSegmentsOnCeilTessFactor = floatToFixed(1.0f / (float)numCeilSegments);
}

static inline FXP PlacePointIn1D(thread const TESS_FACTOR_CONTEXT* TessFactorCtx, bool odd,
                                 int point) {
    bool bFlip = point >= TessFactorCtx->numHalfTessFactorPoints;

    if (bFlip) {
        point = (TessFactorCtx->numHalfTessFactorPoints << 1) - point - odd;
    }

    if (point == TessFactorCtx->numHalfTessFactorPoints)
        return FXP_ONE_HALF;

    unsigned int indexOnCeilHalfTessFactor = point;
    unsigned int indexOnFloorHalfTessFactor = indexOnCeilHalfTessFactor;
    if (point > TessFactorCtx->splitPointOnFloorHalfTessFactor) {
        indexOnFloorHalfTessFactor -= 1;
    }

    FXP fxpLocationOnFloorHalfTessFactor =
        indexOnFloorHalfTessFactor * TessFactorCtx->fxpInvNumSegmentsOnFloorTessFactor;
    FXP fxpLocationOnCeilHalfTessFactor =
        indexOnCeilHalfTessFactor * TessFactorCtx->fxpInvNumSegmentsOnCeilTessFactor;

    FXP fxpLocation =
        fxpLocationOnFloorHalfTessFactor * (FXP_ONE - TessFactorCtx->fxpHalfTessFactorFraction) +
        fxpLocationOnCeilHalfTessFactor * (TessFactorCtx->fxpHalfTessFactorFraction);
    fxpLocation = (fxpLocation + FXP_ONE_HALF) >> FXP_FRACTION_BITS;
    if (bFlip) {
        fxpLocation = FXP_ONE - fxpLocation;
    }
    return fxpLocation;
}

static inline void StitchRegular(thread struct CHWTessellator* ctx, bool bTrapezoid,
                                 DIAGONALS diagonals, int baseIndexOffset, int numInsideEdgePoints,
                                 int insideEdgePointBaseOffset, int outsideEdgePointBaseOffset) {
    int insidePoint = insideEdgePointBaseOffset;
    int outsidePoint = outsideEdgePointBaseOffset;
    if (bTrapezoid) {
        DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint, baseIndexOffset);
        baseIndexOffset += 3;
        outsidePoint++;
    }
    int p;
    switch (diagonals) {
    case DIAGONALS_INSIDE_TO_OUTSIDE:

        for (p = 0; p < numInsideEdgePoints - 1; p++) {
            DefineTriangle(ctx, insidePoint, outsidePoint, outsidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;

            DefineTriangle(ctx, insidePoint, outsidePoint + 1, insidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            insidePoint++;
            outsidePoint++;
        }
        break;
    case DIAGONALS_INSIDE_TO_OUTSIDE_EXCEPT_MIDDLE:

        for (p = 0; p < numInsideEdgePoints / 2 - 1; p++) {
            DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint, baseIndexOffset);
            baseIndexOffset += 3;
            DefineTriangle(ctx, insidePoint, outsidePoint + 1, insidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            insidePoint++;
            outsidePoint++;
        }

        DefineTriangle(ctx, outsidePoint, insidePoint + 1, insidePoint, baseIndexOffset);
        baseIndexOffset += 3;
        DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint + 1, baseIndexOffset);
        baseIndexOffset += 3;
        insidePoint++;
        outsidePoint++;
        p += 2;

        for (; p < numInsideEdgePoints; p++) {
            DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint, baseIndexOffset);
            baseIndexOffset += 3;
            DefineTriangle(ctx, insidePoint, outsidePoint + 1, insidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            insidePoint++;
            outsidePoint++;
        }
        break;
    case DIAGONALS_MIRRORED:

        for (p = 0; p < numInsideEdgePoints / 2; p++) {
            DefineTriangle(ctx, outsidePoint, insidePoint + 1, insidePoint, baseIndexOffset);
            baseIndexOffset += 3;
            DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            insidePoint++;
            outsidePoint++;
        }

        for (; p < numInsideEdgePoints - 1; p++) {
            DefineTriangle(ctx, insidePoint, outsidePoint, outsidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            DefineTriangle(ctx, insidePoint, outsidePoint + 1, insidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            insidePoint++;
            outsidePoint++;
        }
        break;
    }
    if (bTrapezoid) {
        DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint, baseIndexOffset);
        baseIndexOffset += 3;
    }
}

static inline int loop_start(int N) {
    if (N < 2)
        return 1;
    else if (N == 2)
        return 17;
    else if (N < 5)
        return 9;
    else if (N < 9)
        return 5;
    else if (N < 17)
        return 3;
    else
        return 2;
}

static int loop_end(int N) {
    if (N < 2)
        return 0;
    else if (N < 4)
        return 17;
    else if (N < 8)
        return 25;
    else if (N < 16)
        return 29;
    else if (N < 32)
        return 31;
    else
        return 32;
}

static inline uint finalPointPositionTable(uint x) {
    if (x == 0)
        return 0;
    if (x == 1)
        return 0x20;

    uint shift;
    if ((x & 1) == 0) {
        shift = 1;
    } else if ((x & 3) == 3) {
        shift = 2;
    } else if ((x & 7) == 5) {
        shift = 3;
    } else if (x != 17) {
        shift = 4;
    } else {
        shift = 5;
    }

    uint items_to_shift = x | (0x20 << 16);
    uint shifted = items_to_shift >> shift;

    uint bias = min(0xfu, shifted >> 16);
    return bias + (shifted & 0xffff);
}

static inline void StitchTransition(thread struct CHWTessellator* ctx, int baseIndexOffset,
                                    int insideEdgePointBaseOffset,
                                    int insideNumHalfTessFactorPoints, bool insideEdgeTessFactorOdd,
                                    int outsideEdgePointBaseOffset,
                                    int outsideNumHalfTessFactorPoints, bool outsideTessFactorOdd) {
    if (insideEdgeTessFactorOdd) {
        insideNumHalfTessFactorPoints -= 1;
    }
    if (outsideTessFactorOdd) {
        outsideNumHalfTessFactorPoints -= 1;
    }

    int outsidePoint = outsideEdgePointBaseOffset;
    int insidePoint = insideEdgePointBaseOffset;

    int iStart =
        min(loop_start(insideNumHalfTessFactorPoints), loop_start(outsideNumHalfTessFactorPoints));
    int iEnd = loop_end(max(insideNumHalfTessFactorPoints, outsideNumHalfTessFactorPoints));

    if (0 < outsideNumHalfTessFactorPoints) {

        DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint, baseIndexOffset);
        baseIndexOffset += 3;
        outsidePoint++;
    }

    for (int i = iStart; i <= iEnd; i++) {
        int bound = finalPointPositionTable(i);

        if (bound < insideNumHalfTessFactorPoints) {

            DefineTriangle(ctx, insidePoint, outsidePoint, insidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            insidePoint++;
        }
        if (bound < outsideNumHalfTessFactorPoints) {

            DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint, baseIndexOffset);
            baseIndexOffset += 3;
            outsidePoint++;
        }
    }

    if ((insideEdgeTessFactorOdd != outsideTessFactorOdd) || insideEdgeTessFactorOdd) {
        if (insideEdgeTessFactorOdd == outsideTessFactorOdd) {

            DefineTriangle(ctx, insidePoint, outsidePoint, insidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            DefineTriangle(ctx, insidePoint + 1, outsidePoint, outsidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            insidePoint++;
            outsidePoint++;
        } else if (!insideEdgeTessFactorOdd) {

            DefineTriangle(ctx, insidePoint, outsidePoint, outsidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            outsidePoint++;
        } else {

            DefineTriangle(ctx, insidePoint, outsidePoint, insidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            insidePoint++;
        }
    }

    for (int i = iEnd; i >= iStart; i--) {
        int bound = finalPointPositionTable(i);

        if (bound < outsideNumHalfTessFactorPoints) {

            DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint, baseIndexOffset);
            baseIndexOffset += 3;
            outsidePoint++;
        }
        if (bound < insideNumHalfTessFactorPoints) {

            DefineTriangle(ctx, insidePoint, outsidePoint, insidePoint + 1, baseIndexOffset);
            baseIndexOffset += 3;
            insidePoint++;
        }
    }

    if (0 < outsideNumHalfTessFactorPoints) {
        DefineTriangle(ctx, outsidePoint, outsidePoint + 1, insidePoint, baseIndexOffset);
        baseIndexOffset += 3;
        outsidePoint++;
    }
}

static inline void TessellateIsoline(thread const TessParams* params) {
    Partitioning partitioning = params->partitioning;

    bool lineDensityOdd;
    bool lineDetailOdd;
    TESS_FACTOR_CONTEXT lineDensityTessFactorCtx;
    TESS_FACTOR_CONTEXT lineDetailTessFactorCtx;

    thread const float* factors = params->factors;
    float TessFactor_V_LineDensity = factors[0];
    float TessFactor_U_LineDetail = factors[1];

    if (!(TessFactor_V_LineDensity > 0) || !(TessFactor_U_LineDetail > 0)) {
        DrawEmpty(params);
        return;
    }

    TessFactor_V_LineDensity =
        clamp(TessFactor_V_LineDensity, TESS_MIN_ISOLINE_DENSITY_TESSELLATION_FACTOR,
              TESS_MAX_ISOLINE_DENSITY_TESSELLATION_FACTOR);
    TessFactor_U_LineDetail = clamp_factor(TessFactor_U_LineDetail, partitioning, 0);

    if (partitioning == Integer) {
        lineDetailOdd = isOdd(TessFactor_U_LineDetail);
    } else {
        lineDetailOdd = (partitioning == FractionalOdd);
    }

    FXP fxpTessFactor_U_LineDetail = floatToFixed(TessFactor_U_LineDetail);

    ComputeTessFactorCtx(lineDetailOdd, fxpTessFactor_U_LineDetail, &lineDetailTessFactorCtx);
    int numPointsPerLine = NumPointsForTessFactor(lineDetailOdd, fxpTessFactor_U_LineDetail);

    TessFactor_V_LineDensity = ceil(TessFactor_V_LineDensity);
    lineDensityOdd = isOdd(TessFactor_V_LineDensity);
    FXP fxpTessFactor_V_LineDensity = floatToFixed(TessFactor_V_LineDensity);
    ComputeTessFactorCtx(lineDensityOdd, fxpTessFactor_V_LineDensity, &lineDensityTessFactorCtx);

    int numLines = NumPointsForTessFactor(lineDensityOdd, fxpTessFactor_V_LineDensity) - 1;

    uint num_points = numPointsPerLine * numLines;
    {
        AllocatePoints(params, num_points);

        for (int line = 0, pointOffset = 0; line < numLines; line++) {
            FXP fxpV = PlacePointIn1D(&lineDensityTessFactorCtx, lineDensityOdd, line);

            for (int point = 0; point < numPointsPerLine; point++) {
                FXP fxpU = PlacePointIn1D(&lineDetailTessFactorCtx, lineDetailOdd, point);

                DefinePoint(params, pointOffset++, fxpU, fxpV);
            }
        }
    }

    struct CHWTessellator ctx = {
        .params = params,
    };

    if (!params->points_mode) {
        uint num_indices = numLines * (numPointsPerLine - 1) * 2;
        ctx.Index = AllocateIndices(params, num_indices);

        for (int line = 0, pointOffset = 0, indexOffset = 0; line < numLines; line++) {
            pointOffset++;

            for (int point = 1; point < numPointsPerLine; point++) {
                DefineIndex(&ctx, pointOffset - 1, indexOffset++);
                DefineIndex(&ctx, pointOffset, indexOffset++);
                pointOffset++;
            }
        }
    } else {
        DrawPoints(params, num_points);
    }
}

static inline void TessellateTriangle(thread const TessParams* params) {
    Partitioning partitioning = params->partitioning;

    thread const float* factors = params->factors;
    float tessFactor_Ueq0 = factors[0];
    float tessFactor_Veq0 = factors[1];
    float tessFactor_Weq0 = factors[2];
    float insideTessFactor_f = factors[4];

    struct CHWTessellator ctx = {
        .params = params,
    };

    if (!(tessFactor_Ueq0 > 0) || !(tessFactor_Veq0 > 0) || !(tessFactor_Weq0 > 0)) {

        DrawEmpty(params);

        return;
    }

    FXP outsideTessFactor[TRI_EDGES];
    FXP insideTessFactor;
    bool outsideTessFactorOdd[TRI_EDGES];
    bool insideTessFactorOdd;
    TESS_FACTOR_CONTEXT outsideTessFactorCtx[TRI_EDGES];
    TESS_FACTOR_CONTEXT insideTessFactorCtx;

    int numPointsForOutsideEdge[TRI_EDGES];
    int numPointsForInsideTessFactor;
    int insideEdgePointBaseOffset;

    tessFactor_Ueq0 = clamp_factor(tessFactor_Ueq0, partitioning, 0);
    tessFactor_Veq0 = clamp_factor(tessFactor_Veq0, partitioning, 0);
    tessFactor_Weq0 = clamp_factor(tessFactor_Weq0, partitioning, 0);

    float maxf = fmax(fmax(tessFactor_Ueq0, tessFactor_Veq0), tessFactor_Weq0);
    insideTessFactor_f = clamp_factor(insideTessFactor_f, partitioning, maxf);

    float outsideTessFactor_f[TRI_EDGES] = {tessFactor_Ueq0, tessFactor_Veq0, tessFactor_Weq0};
    if (partitioning == Integer) {
        for (int edge = 0; edge < TRI_EDGES; edge++) {
            outsideTessFactorOdd[edge] = isOdd(outsideTessFactor_f[edge]);
        }
        insideTessFactorOdd = isOdd(insideTessFactor_f) && (1.0f != insideTessFactor_f);
    } else {
        bool odd = (partitioning == FractionalOdd);

        for (int edge = 0; edge < TRI_EDGES; edge++) {
            outsideTessFactorOdd[edge] = odd;
        }
        insideTessFactorOdd = odd;
    }

    for (int edge = 0; edge < TRI_EDGES; edge++) {
        outsideTessFactor[edge] = floatToFixed(outsideTessFactor_f[edge]);
    }
    insideTessFactor = floatToFixed(insideTessFactor_f);

    if (partitioning != FractionalEven) {

        if ((FXP_ONE == insideTessFactor) && (FXP_ONE == outsideTessFactor[Ueq0]) &&
            (FXP_ONE == outsideTessFactor[Veq0]) && (FXP_ONE == outsideTessFactor[Weq0])) {

            AllocatePoints(params, 3);

            DefinePoint(params, 0, 0, FXP_ONE);
            DefinePoint(params, 1, 0, 0);
            DefinePoint(params, 2, FXP_ONE, 0);

            if (!params->points_mode) {
                ctx.Index = AllocateIndices(params, 3);

                DefineTriangle(&ctx, 0, 1, 2, 0);
            } else {
                DrawPoints(params, 3);
            }

            return;
        }
    }

    for (int edge = 0; edge < TRI_EDGES; edge++) {
        ComputeTessFactorCtx(outsideTessFactorOdd[edge], outsideTessFactor[edge],
                             &outsideTessFactorCtx[edge]);
    }
    ComputeTessFactorCtx(insideTessFactorOdd, insideTessFactor, &insideTessFactorCtx);

    int NumPoints = 0;

    for (int edge = 0; edge < TRI_EDGES; edge++) {
        numPointsForOutsideEdge[edge] =
            NumPointsForTessFactor(outsideTessFactorOdd[edge], outsideTessFactor[edge]);
        NumPoints += numPointsForOutsideEdge[edge];
    }
    NumPoints -= 3;

    numPointsForInsideTessFactor = NumPointsForTessFactor(insideTessFactorOdd, insideTessFactor);
    {
        int pointCountMin = insideTessFactorOdd ? 4 : 3;

        numPointsForInsideTessFactor = max(pointCountMin, numPointsForInsideTessFactor);
    }

    insideEdgePointBaseOffset = NumPoints;

    {
        int interiorRings = (numPointsForInsideTessFactor >> 1) - 1;
        int even = insideTessFactorOdd ? 0 : 1;
        NumPoints += TRI_EDGES * (interiorRings * (interiorRings + even)) + even;
    }

    {
        AllocatePoints(params, NumPoints);

        int pointOffset = 0;
        for (int edge = 0; edge < TRI_EDGES; edge++) {
            int odd = edge & 0x1;
            int endPoint = numPointsForOutsideEdge[edge] - 1;

            for (int p = 0; p < endPoint; p++, pointOffset++) {

                int q = odd ? p : endPoint - p;

                FXP fxpParam =
                    PlacePointIn1D(&outsideTessFactorCtx[edge], outsideTessFactorOdd[edge], q);
                DefinePoint(params, pointOffset, (edge == 0) ? 0 : fxpParam,
                            (edge == 0)   ? fxpParam
                            : (edge == 2) ? FXP_ONE - fxpParam
                                          : 0);
            }
        }

        int numRings = (numPointsForInsideTessFactor >> 1);
        for (int ring = 1; ring < numRings; ring++) {
            int startPoint = ring;
            int endPoint = numPointsForInsideTessFactor - 1 - startPoint;

            int perpendicularAxisPoint = startPoint;
            FXP fxpPerpParam =
                PlacePointIn1D(&insideTessFactorCtx, insideTessFactorOdd, perpendicularAxisPoint);

            fxpPerpParam *= FXP_TWO_THIRDS;
            fxpPerpParam = (fxpPerpParam + FXP_ONE_HALF) >> FXP_FRACTION_BITS;

            for (int edge = 0; edge < TRI_EDGES; edge++) {
                int odd = edge & 0x1;

                for (int p = startPoint; p < endPoint; p++, pointOffset++) {

                    int q = odd ? p : endPoint - (p - startPoint);

                    FXP fxpParam = PlacePointIn1D(&insideTessFactorCtx, insideTessFactorOdd, q);

                    const unsigned int deriv = 2;

                    FXP tmp = fxpParam - (fxpPerpParam + 1) / deriv;

                    DefinePoint(params, pointOffset, edge > 0 ? tmp : fxpPerpParam,
                                edge == 0   ? tmp
                                : edge == 1 ? fxpPerpParam
                                            : FXP_ONE - tmp - fxpPerpParam);
                }
            }
        }
        if (!insideTessFactorOdd) {

            DefinePoint(params, pointOffset, FXP_ONE_THIRD, FXP_ONE_THIRD);
        }
    }

    if (params->points_mode) {
        DrawPoints(params, NumPoints);
        return;
    }

    {

        int numRings = ((numPointsForInsideTessFactor + 1) >> 1);

        int NumIndices = 0;
        {
            int OuterPoints = numPointsForOutsideEdge[0] + numPointsForOutsideEdge[1] +
                              numPointsForOutsideEdge[2];

            int numRings18 = numRings * 18;
            NumIndices = ((numRings18 - 27) * numPointsForInsideTessFactor) + (3 * OuterPoints) -
                         (numRings18 * (numRings - 1)) + (insideTessFactorOdd ? 3 : 0);
        }

        ctx.Index = AllocateIndices(params, NumIndices);

        int insideOffset = insideEdgePointBaseOffset;
        int outsideEdgePointBaseOffset = 0;

        NumIndices = 0;
        for (int ring = 1; ring < numRings; ring++) {
            int numPointsForInsideEdge = numPointsForInsideTessFactor - 2 * ring;
            int edge0InsidePointBaseOffset = insideOffset;
            int edge0OutsidePointBaseOffset = outsideEdgePointBaseOffset;
            for (int edge = 0; edge < TRI_EDGES; edge++) {
                int outsidePoints =
                    ring == 1 ? numPointsForOutsideEdge[edge] : (numPointsForInsideEdge + 2);

                int numTriangles = numPointsForInsideEdge + outsidePoints - 2;

                int insideBaseOffset;
                int outsideBaseOffset;
                if (edge == 2) {
                    ctx.IndexPatchCtx.insidePointIndexDeltaToRealValue = insideOffset;
                    ctx.IndexPatchCtx.insidePointIndexBadValue = numPointsForInsideEdge - 1;
                    ctx.IndexPatchCtx.insidePointIndexReplacementValue = edge0InsidePointBaseOffset;
                    ctx.IndexPatchCtx.outsidePointIndexPatchBase =
                        ctx.IndexPatchCtx.insidePointIndexBadValue + 1;
                    ctx.IndexPatchCtx.outsidePointIndexDeltaToRealValue =
                        outsideEdgePointBaseOffset - ctx.IndexPatchCtx.outsidePointIndexPatchBase;
                    ctx.IndexPatchCtx.outsidePointIndexBadValue =
                        ctx.IndexPatchCtx.outsidePointIndexPatchBase + outsidePoints - 1;
                    ctx.IndexPatchCtx.outsidePointIndexReplacementValue =
                        edge0OutsidePointBaseOffset;
                    ctx.bUsingPatchedIndices = true;
                    insideBaseOffset = 0;
                    outsideBaseOffset = ctx.IndexPatchCtx.outsidePointIndexPatchBase;
                } else {
                    insideBaseOffset = insideOffset;
                    outsideBaseOffset = outsideEdgePointBaseOffset;
                }
                if (ring == 1) {
                    StitchTransition(&ctx, NumIndices, insideBaseOffset,
                                     insideTessFactorCtx.numHalfTessFactorPoints,
                                     insideTessFactorOdd, outsideBaseOffset,
                                     outsideTessFactorCtx[edge].numHalfTessFactorPoints,
                                     outsideTessFactorOdd[edge]);
                } else {
                    StitchRegular(&ctx, true, DIAGONALS_MIRRORED, NumIndices,
                                  numPointsForInsideEdge, insideBaseOffset, outsideBaseOffset);
                }
                if (2 == edge) {
                    ctx.bUsingPatchedIndices = false;
                }
                NumIndices += numTriangles * 3;
                outsideEdgePointBaseOffset += outsidePoints - 1;
                insideOffset += numPointsForInsideEdge - 1;
            }
        }
        if (insideTessFactorOdd) {

            DefineTriangle(&ctx, outsideEdgePointBaseOffset, outsideEdgePointBaseOffset + 1,
                           outsideEdgePointBaseOffset + 2, NumIndices);
            NumIndices += 3;
        }
    }
}

static inline void TessellateQuad(thread const TessParams* params) {
    Partitioning partitioning = params->partitioning;
    thread const float* factors = params->factors;

    float tessFactor_Ueq0 = factors[0];
    float tessFactor_Veq0 = factors[1];
    float tessFactor_Ueq1 = factors[2];
    float tessFactor_Veq1 = factors[3];

    float insideTessFactor_U = factors[4];
    float insideTessFactor_V = factors[5];

    struct CHWTessellator ctx = {
        .params = params,
    };

    if (!(tessFactor_Ueq0 > 0) || !(tessFactor_Veq0 > 0) || !(tessFactor_Ueq1 > 0) ||
        !(tessFactor_Veq1 > 0)) {
        DrawEmpty(params);
        return;
    }

    FXP outsideTessFactor[QUAD_EDGES];
    FXP insideTessFactor[QUAD_AXES];
    bool outsideTessFactorOdd[QUAD_EDGES];
    bool insideTessFactorOdd[QUAD_AXES];
    TESS_FACTOR_CONTEXT outsideTessFactorCtx[QUAD_EDGES];
    TESS_FACTOR_CONTEXT insideTessFactorCtx[QUAD_AXES];

    int numPointsForOutsideEdge[QUAD_EDGES];
    int numPointsForInsideTessFactor[QUAD_AXES];
    int insideEdgePointBaseOffset;

    tessFactor_Ueq0 = clamp_factor(tessFactor_Ueq0, partitioning, 0);
    tessFactor_Veq0 = clamp_factor(tessFactor_Veq0, partitioning, 0);
    tessFactor_Ueq1 = clamp_factor(tessFactor_Ueq1, partitioning, 0);
    tessFactor_Veq1 = clamp_factor(tessFactor_Veq1, partitioning, 0);

    float maxf =
        fmax(fmax(fmax(tessFactor_Ueq0, tessFactor_Veq0), fmax(tessFactor_Ueq1, tessFactor_Veq1)),
             fmax(insideTessFactor_U, insideTessFactor_V));

    insideTessFactor_U = clamp_factor(insideTessFactor_U, partitioning, maxf);
    insideTessFactor_V = clamp_factor(insideTessFactor_V, partitioning, maxf);

    float outsideTessFactor_f[QUAD_EDGES] = {tessFactor_Ueq0, tessFactor_Veq0, tessFactor_Ueq1,
                                             tessFactor_Veq1};
    float insideTessFactor_f[QUAD_AXES] = {insideTessFactor_U, insideTessFactor_V};
    if (partitioning == Integer) {
        for (int edge = 0; edge < QUAD_EDGES; edge++) {
            outsideTessFactorOdd[edge] = isOdd(outsideTessFactor_f[edge]);
        }
        for (int axis = 0; axis < QUAD_AXES; axis++) {
            insideTessFactorOdd[axis] =
                isOdd(insideTessFactor_f[axis]) && (1.0f != insideTessFactor_f[axis]);
        }
    } else {
        bool odd = (partitioning == FractionalOdd);

        for (int edge = 0; edge < QUAD_EDGES; edge++) {
            outsideTessFactorOdd[edge] = odd;
        }
        insideTessFactorOdd[U] = insideTessFactorOdd[V] = odd;
    }

    for (int edge = 0; edge < QUAD_EDGES; edge++) {
        outsideTessFactor[edge] = floatToFixed(outsideTessFactor_f[edge]);
    }
    for (int axis = 0; axis < QUAD_AXES; axis++) {
        insideTessFactor[axis] = floatToFixed(insideTessFactor_f[axis]);
    }

    if (partitioning != FractionalEven) {

        if ((FXP_ONE == insideTessFactor[U]) && (FXP_ONE == insideTessFactor[V]) &&
            (FXP_ONE == outsideTessFactor[Ueq0]) && (FXP_ONE == outsideTessFactor[Veq0]) &&
            (FXP_ONE == outsideTessFactor[Ueq1]) && (FXP_ONE == outsideTessFactor[Veq1])) {

            if (!params->points_mode) {
                ctx.Index = AllocateIndices(params, 6);

                DefineTriangle(&ctx, 0, 1, 3, 0);
                DefineTriangle(&ctx, 1, 2, 3, 3);
            } else {
                DrawPoints(params, 4);
            }

            AllocatePoints(params, 4);

            DefinePoint(params, 0, 0, 0);
            DefinePoint(params, 1, FXP_ONE, 0);
            DefinePoint(params, 2, FXP_ONE, FXP_ONE);
            DefinePoint(params, 3, 0, FXP_ONE);
            return;
        }
    }

    for (int edge = 0; edge < QUAD_EDGES; edge++) {
        ComputeTessFactorCtx(outsideTessFactorOdd[edge], outsideTessFactor[edge],
                             &outsideTessFactorCtx[edge]);
    }

    for (int axis = 0; axis < QUAD_AXES; axis++) {
        ComputeTessFactorCtx(insideTessFactorOdd[axis], insideTessFactor[axis],
                             &insideTessFactorCtx[axis]);
    }

    int NumPoints = 0;

    for (int edge = 0; edge < QUAD_EDGES; edge++) {
        numPointsForOutsideEdge[edge] =
            NumPointsForTessFactor(outsideTessFactorOdd[edge], outsideTessFactor[edge]);
        NumPoints += numPointsForOutsideEdge[edge];
    }
    NumPoints -= 4;

    for (int axis = 0; axis < QUAD_AXES; axis++) {
        numPointsForInsideTessFactor[axis] =
            NumPointsForTessFactor(insideTessFactorOdd[axis], insideTessFactor[axis]);
        int pointCountMin = insideTessFactorOdd[axis] ? 4 : 3;

        numPointsForInsideTessFactor[axis] = max(pointCountMin, numPointsForInsideTessFactor[axis]);
    }

    insideEdgePointBaseOffset = NumPoints;

    int numInteriorPoints =
        (numPointsForInsideTessFactor[U] - 2) * (numPointsForInsideTessFactor[V] - 2);
    NumPoints += numInteriorPoints;

    {
        AllocatePoints(params, NumPoints);

        int pointOffset = 0;
        for (int edge = 0; edge < QUAD_EDGES; edge++) {
            int odd = edge & 0x1;

            int endPoint = numPointsForOutsideEdge[edge] - 1;
            for (int p = 0; p < endPoint; p++, pointOffset++) {
                int q = ((edge == 1) || (edge == 2)) ? p : endPoint - p;
                FXP fxpParam =
                    PlacePointIn1D(&outsideTessFactorCtx[edge], outsideTessFactorOdd[edge], q);

                FXP u = odd ? fxpParam : ((edge == 2) ? FXP_ONE : 0);
                FXP v = odd ? ((edge == 3) ? FXP_ONE : 0) : fxpParam;
                DefinePoint(params, pointOffset, u, v);
            }
        }

        int minNumPointsForTessFactor =
            min(numPointsForInsideTessFactor[U], numPointsForInsideTessFactor[V]);

        int numRings = (minNumPointsForTessFactor >> 1);

        for (int ring = 1; ring < numRings; ring++) {
            int startPoint = ring;
            int endPoint[QUAD_AXES] = {
                numPointsForInsideTessFactor[U] - 1 - startPoint,
                numPointsForInsideTessFactor[V] - 1 - startPoint,
            };

            for (int edge = 0; edge < QUAD_EDGES; edge++) {
                int odd[QUAD_AXES] = {edge & 0x1, ((edge + 1) & 0x1)};
                int perpendicularAxisPoint = (edge < 2) ? startPoint : endPoint[odd[0]];
                FXP fxpPerpParam =
                    PlacePointIn1D(&insideTessFactorCtx[odd[0]], insideTessFactorOdd[odd[0]],
                                   perpendicularAxisPoint);

                for (int p = startPoint; p < endPoint[odd[1]]; p++, pointOffset++)

                {
                    bool odd_ = odd[1];
                    int q = ((edge == 1) || (edge == 2)) ? p : endPoint[odd_] - (p - startPoint);
                    FXP fxpParam =
                        PlacePointIn1D(&insideTessFactorCtx[odd_], insideTessFactorOdd[odd_], q);
                    DefinePoint(params, pointOffset, odd_ ? fxpPerpParam : fxpParam,
                                odd_ ? fxpParam : fxpPerpParam);
                }
            }
        }

        if ((numPointsForInsideTessFactor[U] > numPointsForInsideTessFactor[V]) &&
            !insideTessFactorOdd[V]) {
            int startPoint = numRings;
            int endPoint = numPointsForInsideTessFactor[U] - 1 - startPoint;
            for (int p = startPoint; p <= endPoint; p++, pointOffset++) {
                FXP fxpParam = PlacePointIn1D(&insideTessFactorCtx[U], insideTessFactorOdd[U], p);
                DefinePoint(params, pointOffset, fxpParam, FXP_ONE_HALF);
            }
        } else if ((numPointsForInsideTessFactor[V] >= numPointsForInsideTessFactor[U]) &&
                   !insideTessFactorOdd[U]) {
            int startPoint = numRings;
            int endPoint = numPointsForInsideTessFactor[V] - 1 - startPoint;
            for (int p = endPoint; p >= startPoint; p--, pointOffset++) {
                FXP fxpParam = PlacePointIn1D(&insideTessFactorCtx[V], insideTessFactorOdd[V], p);
                DefinePoint(params, pointOffset, FXP_ONE_HALF, fxpParam);
            }
        }
    }

    if (params->points_mode) {
        DrawPoints(params, NumPoints);
        return;
    }

    {

        int numPointRowsToCenter[QUAD_AXES] = {
            (numPointsForInsideTessFactor[U] + 1) >> 1,
            (numPointsForInsideTessFactor[V] + 1) >> 1,
        };

        int numRings = min(numPointRowsToCenter[U], numPointRowsToCenter[V]);

        {

            int OuterPoints = numPointsForOutsideEdge[0] + numPointsForOutsideEdge[1] +
                              numPointsForOutsideEdge[2] + numPointsForOutsideEdge[3];

            int InnerPoints = numPointsForInsideTessFactor[U] + numPointsForInsideTessFactor[V];

            int NumIndices = (OuterPoints * 3) + (12 * numRings * InnerPoints) -
                             (InnerPoints * 18) - (24 * numRings * (numRings - 1));

            bool U_major = (numPointsForInsideTessFactor[U] > numPointsForInsideTessFactor[V]);
            unsigned M = U_major ? U : V;
            unsigned m = U_major ? V : U;

            if (insideTessFactorOdd[m]) {
                NumIndices += 12 * ((numPointsForInsideTessFactor[M] >> 1) -
                                    (numPointsForInsideTessFactor[m] >> 1));
                NumIndices += (insideTessFactorOdd[M] ? 6 : 12);
            }

            ctx.Index = AllocateIndices(params, NumIndices);
        }

        int degeneratePointRing[QUAD_AXES] = {

            !insideTessFactorOdd[V] ? numPointRowsToCenter[V] - 1 : -1,
            !insideTessFactorOdd[U] ? numPointRowsToCenter[U] - 1 : -1,
        };

        int numPointsForOutsideEdge_[QUAD_EDGES] = {
            numPointsForOutsideEdge[Ueq0],
            numPointsForOutsideEdge[Veq0],
            numPointsForOutsideEdge[Ueq1],
            numPointsForOutsideEdge[Veq1],
        };

        int insideEdgePointBaseOffset_ = insideEdgePointBaseOffset;
        int outsideEdgePointBaseOffset = 0;

        int NumIndices = 0;

        for (int ring = 1; ring < numRings; ring++) {
            int numPointsForInsideEdge[QUAD_AXES] = {numPointsForInsideTessFactor[U] - 2 * ring,
                                                     numPointsForInsideTessFactor[V] - 2 * ring};

            int edge0InsidePointBaseOffset = insideEdgePointBaseOffset_;
            int edge0OutsidePointBaseOffset = outsideEdgePointBaseOffset;

            for (int edge = 0; edge < QUAD_EDGES; edge++) {
                int odd = (edge + 1) & 0x1;

                int numTriangles = numPointsForInsideEdge[odd] + numPointsForOutsideEdge_[edge] - 2;
                int insideBaseOffset;
                int outsideBaseOffset;

                if (edge == 3) {
                    if (ring == degeneratePointRing[odd]) {
                        ctx.IndexPatchCtx2.baseIndexToInvert = insideEdgePointBaseOffset_ + 1;
                        ctx.IndexPatchCtx2.cornerCaseBadValue =
                            outsideEdgePointBaseOffset + numPointsForOutsideEdge_[edge] - 1;
                        ctx.IndexPatchCtx2.cornerCaseReplacementValue = edge0OutsidePointBaseOffset;
                        ctx.IndexPatchCtx2.indexInversionEndPoint =
                            (ctx.IndexPatchCtx2.baseIndexToInvert << 1) - 1;
                        insideBaseOffset = ctx.IndexPatchCtx2.baseIndexToInvert;
                        outsideBaseOffset = outsideEdgePointBaseOffset;
                        ctx.bUsingPatchedIndices2 = true;
                    } else {
                        ctx.IndexPatchCtx.insidePointIndexDeltaToRealValue =
                            insideEdgePointBaseOffset_;
                        ctx.IndexPatchCtx.insidePointIndexBadValue =
                            numPointsForInsideEdge[odd] - 1;
                        ctx.IndexPatchCtx.insidePointIndexReplacementValue =
                            edge0InsidePointBaseOffset;
                        ctx.IndexPatchCtx.outsidePointIndexPatchBase =
                            ctx.IndexPatchCtx.insidePointIndexBadValue + 1;
                        ctx.IndexPatchCtx.outsidePointIndexDeltaToRealValue =
                            outsideEdgePointBaseOffset -
                            ctx.IndexPatchCtx.outsidePointIndexPatchBase;
                        ctx.IndexPatchCtx.outsidePointIndexBadValue =
                            ctx.IndexPatchCtx.outsidePointIndexPatchBase +
                            numPointsForOutsideEdge_[edge] - 1;
                        ctx.IndexPatchCtx.outsidePointIndexReplacementValue =
                            edge0OutsidePointBaseOffset;

                        insideBaseOffset = 0;
                        outsideBaseOffset = ctx.IndexPatchCtx.outsidePointIndexPatchBase;
                        ctx.bUsingPatchedIndices = true;
                    }
                } else if ((edge == 2) && (ring == degeneratePointRing[odd])) {
                    ctx.IndexPatchCtx2.baseIndexToInvert = insideEdgePointBaseOffset_;
                    ctx.IndexPatchCtx2.cornerCaseBadValue = -1;
                    ctx.IndexPatchCtx2.cornerCaseReplacementValue = -1;
                    ctx.IndexPatchCtx2.indexInversionEndPoint = ctx.IndexPatchCtx2.baseIndexToInvert
                                                                << 1;
                    insideBaseOffset = ctx.IndexPatchCtx2.baseIndexToInvert;
                    outsideBaseOffset = outsideEdgePointBaseOffset;
                    ctx.bUsingPatchedIndices2 = true;
                } else {
                    insideBaseOffset = insideEdgePointBaseOffset_;
                    outsideBaseOffset = outsideEdgePointBaseOffset;
                }
                if (ring == 1) {
                    StitchTransition(&ctx, NumIndices, insideBaseOffset,
                                     insideTessFactorCtx[odd].numHalfTessFactorPoints,
                                     insideTessFactorOdd[odd], outsideBaseOffset,
                                     outsideTessFactorCtx[edge].numHalfTessFactorPoints,
                                     outsideTessFactorOdd[edge]);
                } else {
                    StitchRegular(&ctx, true, DIAGONALS_MIRRORED, NumIndices,
                                  numPointsForInsideEdge[odd], insideBaseOffset, outsideBaseOffset);
                }
                ctx.bUsingPatchedIndices = false;
                ctx.bUsingPatchedIndices2 = false;
                NumIndices += numTriangles * 3;
                outsideEdgePointBaseOffset += numPointsForOutsideEdge_[edge] - 1;
                if ((edge == 2) && (ring == degeneratePointRing[odd])) {
                    insideEdgePointBaseOffset_ -= numPointsForInsideEdge[odd] - 1;
                } else {
                    insideEdgePointBaseOffset_ += numPointsForInsideEdge[odd] - 1;
                }
                numPointsForOutsideEdge_[edge] = numPointsForInsideEdge[odd];
            }
        }

        if ((numPointsForInsideTessFactor[U] > numPointsForInsideTessFactor[V]) &&
            insideTessFactorOdd[V]) {
            ctx.bUsingPatchedIndices2 = true;
            int stripNumQuads =
                (((numPointsForInsideTessFactor[U] >> 1) - (numPointsForInsideTessFactor[V] >> 1))
                 << 1) +
                (insideTessFactorOdd[U] ? 1 : 2);
            ctx.IndexPatchCtx2.baseIndexToInvert = outsideEdgePointBaseOffset + stripNumQuads + 2;
            ctx.IndexPatchCtx2.cornerCaseBadValue = ctx.IndexPatchCtx2.baseIndexToInvert;
            ctx.IndexPatchCtx2.cornerCaseReplacementValue = outsideEdgePointBaseOffset;
            ctx.IndexPatchCtx2.indexInversionEndPoint = ctx.IndexPatchCtx2.baseIndexToInvert +
                                                        ctx.IndexPatchCtx2.baseIndexToInvert +
                                                        stripNumQuads;
            StitchRegular(&ctx, false, DIAGONALS_INSIDE_TO_OUTSIDE, NumIndices, stripNumQuads + 1,
                          ctx.IndexPatchCtx2.baseIndexToInvert, outsideEdgePointBaseOffset + 1);
            ctx.bUsingPatchedIndices2 = false;
            NumIndices += stripNumQuads * 6;
        } else if ((numPointsForInsideTessFactor[V] >= numPointsForInsideTessFactor[U]) &&
                   insideTessFactorOdd[U]) {
            ctx.bUsingPatchedIndices2 = true;
            int stripNumQuads =
                (((numPointsForInsideTessFactor[V] >> 1) - (numPointsForInsideTessFactor[U] >> 1))
                 << 1) +
                (insideTessFactorOdd[V] ? 1 : 2);
            ctx.IndexPatchCtx2.baseIndexToInvert = outsideEdgePointBaseOffset + stripNumQuads + 1;
            ctx.IndexPatchCtx2.cornerCaseBadValue = -1;
            ctx.IndexPatchCtx2.indexInversionEndPoint = ctx.IndexPatchCtx2.baseIndexToInvert +
                                                        ctx.IndexPatchCtx2.baseIndexToInvert +
                                                        stripNumQuads;
            DIAGONALS diag = insideTessFactorOdd[V] ? DIAGONALS_INSIDE_TO_OUTSIDE_EXCEPT_MIDDLE
                                                    : DIAGONALS_INSIDE_TO_OUTSIDE;
            StitchRegular(&ctx, false, diag, NumIndices, stripNumQuads + 1,
                          ctx.IndexPatchCtx2.baseIndexToInvert, outsideEdgePointBaseOffset);
            ctx.bUsingPatchedIndices2 = false;
            NumIndices += stripNumQuads * 6;
        }
    }
}

float power_of_two_factor(float value) {
    uint bits = as_type<uint>(fmin(fmax(value, 1.0f), 64.0f));
    if (bits & 0x007fffffu)
        bits = (bits & 0x7f800000u) + 0x00800000u;
    return as_type<float>(bits);
}
float clamp_guest_factor(float value, constant Parameters& config) {
    return value > 0.0f ? fmin(fmax(value, config.minimum_factor), config.maximum_factor) : value;
}
kernel void generate_tessellation_domains(
    device const PatchFactors* source [[buffer(0)]], device DomainPoint* points [[buffer(1)]],
    device uint* indices [[buffer(2)]], device DomainPatch* patches [[buffer(3)]],
    device atomic_uint* errors [[buffer(4)]], constant Parameters& config [[buffer(5)]],
    device atomic_uint* source_errors [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]], uint lane [[thread_index_in_threadgroup]]) {
    const ulong index = (ulong(group.y) * grid.x + group.x) * 32u + lane;
    if (index >= config.patches)
        return;
    const PatchFactors factors = source[index];
    TessStats stats = {};
    stats.metadata.instance = factors.instance;
    stats.metadata.patch = factors.patch;
    TessParams params = {};
    params.points = points + index * MaxPoints;
    params.indices = indices + index * MaxIndices;
    params.stats = &stats;
    params.partitioning = config.partitioning == 2   ? FractionalOdd
                          : config.partitioning == 3 ? FractionalEven
                                                     : Integer;
    params.points_mode = config.topology == 0;
    params.ccw = config.topology == 3;
    params.triangle = config.type == 1;
    for (uint i = 0; i < 4; ++i)
        params.factors[i] = clamp_guest_factor(as_type<float>(factors.outer[i]), config);
    for (uint i = 0; i < 2; ++i)
        params.factors[i + 4] = clamp_guest_factor(as_type<float>(factors.inner[i]), config);
    if (config.partitioning == 1) {
        const uint outer = config.type == 0 ? 2u : config.type == 1 ? 3u : 4u;
        const uint inner = config.type == 0 ? 0u : config.type == 1 ? 1u : 2u;
        for (uint i = config.type == 0 ? 1u : 0u; i < outer; ++i) {
            if (params.factors[i] > 0.0f)
                params.factors[i] = power_of_two_factor(params.factors[i]);
        }
        for (uint i = 0; i < inner; ++i)
            params.factors[4 + i] = power_of_two_factor(params.factors[4 + i]);
    }
    if (atomic_load_explicit(source_errors, memory_order_relaxed)) {
        stats.errors |= 16u;
    } else if (config.type == 0) {
        TessellateIsoline(&params);
    } else if (config.type == 1) {
        TessellateTriangle(&params);
    } else {
        TessellateQuad(&params);
    }
    if (stats.written_points != stats.metadata.point_count ||
        stats.written_indices != stats.metadata.index_count ||
        (stats.metadata.index_count && stats.max_index >= stats.metadata.point_count)) {
        stats.errors |= 8u;
    }
    if (stats.errors) {
        stats.metadata.point_count = 0;
        stats.metadata.index_count = 0;
        atomic_fetch_or_explicit(errors, stats.errors, memory_order_relaxed);
    }
    patches[index] = stats.metadata;
}
)metal";

} // namespace Metal

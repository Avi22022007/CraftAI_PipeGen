// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[ParseConfigDetPgie] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Output tensor shape (batch stripped): [84, 8400]
    // inferDims: numDims=2, d[0]=84, d[1]=8400
    // Layout: 84 rows x 8400 columns — each column is one detection proposal.
    // First 4 rows: cx, cy, w, h (in network-input pixel coords).
    // Rows 4..83: class scores (post-sigmoid probabilities).

    const unsigned int featureDim = layer0.inferDims.d[0]; // 84
    const unsigned int numProposals = layer0.inferDims.d[1]; // 8400

    // Sanity: featureDim should be 4 + numClasses
    if (featureDim != 4u + numClasses) {
        // Try transposed layout: d[0]=numProposals, d[1]=featureDim
        // We handle whichever layout we get.
    }

    // Determine if the tensor is [feat, proposals] or [proposals, feat].
    bool transposed = false; // true means data is [proposals, feat] already
    unsigned int actualFeat = featureDim;
    unsigned int actualCount = numProposals;
    if (featureDim == 4u + numClasses) {
        // [84, 8400] — columns are proposals
        transposed = false;
    } else if (numProposals == 4u + numClasses) {
        // [8400, 84] — rows are proposals
        transposed = true;
        actualFeat = numProposals;
        actualCount = featureDim;
    } else {
        fprintf(stderr, "[ParseConfigDetPgie] unexpected dims: d[0]=%u d[1]=%u numClasses=%u\n",
                featureDim, numProposals, numClasses);
        return false;
    }

    for (unsigned int i = 0; i < actualCount; ++i) {
        // Find the best class score for this proposal
        float maxScore = 0.0f;
        unsigned int bestClass = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float score;
            if (!transposed) {
                // [feat, count]: element at row (4+c), col i
                score = data[(4u + c) * actualCount + i];
            } else {
                // [count, feat]: element at row i, col (4+c)
                score = data[i * actualFeat + 4u + c];
            }
            if (score > maxScore) {
                maxScore = score;
                bestClass = c;
            }
        }

        // Per-class threshold
        float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClass]
                     : 0.35f;
        if (maxScore < thr) continue;

        // Extract cx, cy, w, h
        float cx, cy, bw, bh;
        if (!transposed) {
            cx = data[0u * actualCount + i];
            cy = data[1u * actualCount + i];
            bw = data[2u * actualCount + i];
            bh = data[3u * actualCount + i];
        } else {
            cx = data[i * actualFeat + 0u];
            cy = data[i * actualFeat + 1u];
            bw = data[i * actualFeat + 2u];
            bh = data[i * actualFeat + 3u];
        }

        // Convert centre+size to top-left corner, in network-input coords
        float left   = cx - bw * 0.5f;
        float top    = cy - bh * 0.5f;

        // Clamp to network input
        left  = std::max<float>(0.0f, std::min<float>(left, netW));
        top   = std::max<float>(0.0f, std::min<float>(top, netH));
        float right  = std::max<float>(0.0f, std::min<float>(cx + bw * 0.5f, netW));
        float bottom = std::max<float>(0.0f, std::min<float>(cy + bh * 0.5f, netH));

        float finalW = right - left;
        float finalH = bottom - top;

        // Drop degenerate boxes
        if (finalW <= 1.0f || finalH <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = finalW;
        obj.height = finalH;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}
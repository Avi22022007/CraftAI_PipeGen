#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <map>
#include <string>
#include <algorithm>
#include <deque>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── Per-track persistent state ──────────────────────────────────────────────
struct TrackState {
    bool was_inside;
    std::deque<cv::Point> trail;  // last N center-bottom positions
};

// We use function-local statics wrapped in a struct to keep state across frames.
// The job notes say "static map<int,bool>" — we extend to keep trails too.
struct PersistentState {
    std::map<int, TrackState> tracks;
    int entry_total;
    PersistentState() : entry_total(0) {}
};

static PersistentState &get_state() {
    static PersistentState s;
    return s;
}

// Reference polygon vertices in 2116x1080 frame coordinates (clockwise from bottom-left)
static const float BASE_POLY_X[] = { 528.0f, 752.0f, 1673.0f, 1707.0f };
static const float BASE_POLY_Y[] = { 792.0f, 426.0f,  480.0f,  885.0f };
static const int   BASE_W = 2116;
static const int   BASE_H = 1080;
static const int   NPOLY  = 4;
static const int   MAX_TRAIL = 30;

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[zone_render] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    PersistentState &ps = get_state();

    // Scale polygon to current frame dimensions
    const float sx = (float)W / (float)BASE_W;
    const float sy = (float)H / (float)BASE_H;

    std::vector<cv::Point2f> poly_f(NPOLY);
    std::vector<cv::Point>   poly_i(NPOLY);
    for (int i = 0; i < NPOLY; ++i) {
        poly_f[i] = cv::Point2f(BASE_POLY_X[i] * sx, BASE_POLY_Y[i] * sy);
        poly_i[i] = cv::Point((int)std::round(BASE_POLY_X[i] * sx),
                              (int)std::round(BASE_POLY_Y[i] * sy));
    }

    // Build contour for pointPolygonTest (needs [N,1,2] float32)
    cv::Mat contour(NPOLY, 1, CV_32FC2);
    for (int i = 0; i < NPOLY; ++i) {
        contour.at<cv::Vec2f>(i, 0) = cv::Vec2f(poly_f[i].x, poly_f[i].y);
    }

    // Gather tracked person objects from NvDsObjectMeta
    NvDsFrameMeta *fm = (NvDsFrameMeta *) in->frame_meta;

    struct PersonTrack {
        int track_id;
        float x1, y1, x2, y2;
        float score;
        bool inside;
    };
    std::vector<PersonTrack> persons;

    int inside_count = 0;

    if (fm) {
        for (NvDsMetaList *ol = fm->obj_meta_list; ol; ol = ol->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) ol->data;
            if (!obj) continue;
            // class_id 0 = person for COCO YOLO
            if (obj->class_id != 0) continue;

            float x1 = obj->rect_params.left;
            float y1 = obj->rect_params.top;
            float bw = obj->rect_params.width;
            float bh = obj->rect_params.height;
            float x2 = x1 + bw;
            float y2 = y1 + bh;

            int tid = (int) obj->object_id;
            float feet_x = (x1 + x2) * 0.5f;
            float feet_y = y2;

            // Point-in-polygon test
            double dist = cv::pointPolygonTest(contour, cv::Point2f(feet_x, feet_y), false);
            bool inside = (dist >= 0.0);

            // Transition detection
            bool previously_inside = false;
            auto it = ps.tracks.find(tid);
            if (it != ps.tracks.end()) {
                previously_inside = it->second.was_inside;
            }

            if (inside && !previously_inside) {
                ps.entry_total++;
            }

            // Update persistent state
            TrackState &ts = ps.tracks[tid];
            ts.was_inside = inside;
            // Add to trail (bottom-center)
            ts.trail.push_back(cv::Point((int)std::round(feet_x), (int)std::round(feet_y)));
            if ((int)ts.trail.size() > MAX_TRAIL) {
                ts.trail.pop_front();
            }

            if (inside) inside_count++;

            PersonTrack pt;
            pt.track_id = tid;
            pt.x1 = x1; pt.y1 = y1; pt.x2 = x2; pt.y2 = y2;
            pt.score = obj->confidence;
            pt.inside = inside;
            persons.push_back(pt);
        }
    }

    int person_count = (int)persons.size();

    // ── DRAW ────────────────────────────────────────────────────────────────
    cv::Mat frame = read_nv12_bgr(in);

    const float scale = (float)W / 1400.0f;
    const int box_thick   = std::max(2, (int)(3.0f * scale));
    const double label_scale = std::max(0.8, 1.1 * (double)scale);
    const int label_thick  = std::max(1, (int)(2.0f * scale));
    const int zone_thick   = std::max(2, (int)(4.0f * scale));

    // Draw hazard-zone polygon outline in red
    {
        const cv::Point *pts_arr = poly_i.data();
        int npts = NPOLY;
        cv::polylines(frame, &pts_arr, &npts, 1, true, cv::Scalar(0, 0, 255), zone_thick, cv::LINE_AA);
    }

    // Draw each person
    for (size_t pi = 0; pi < persons.size(); ++pi) {
        const PersonTrack &p = persons[pi];
        cv::Scalar color = p.inside ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 200, 0);

        int ix1 = (int)std::round(p.x1), iy1 = (int)std::round(p.y1);
        int ix2 = (int)std::round(p.x2), iy2 = (int)std::round(p.y2);

        // Bounding box
        cv::rectangle(frame, cv::Point(ix1, iy1), cv::Point(ix2, iy2), color, box_thick);

        // Trail
        auto tit = ps.tracks.find(p.track_id);
        if (tit != ps.tracks.end()) {
            const std::deque<cv::Point> &trail = tit->second.trail;
            int trail_thick = std::max(1, box_thick / 2);
            for (int ti = 1; ti < (int)trail.size(); ++ti) {
                cv::line(frame, trail[(size_t)(ti - 1)], trail[(size_t)ti], color, trail_thick, cv::LINE_AA);
            }
        }

        // Label: "ID <N> IN ZONE" or "ID <N> CLEAR"
        char label_buf[64];
        if (p.inside)
            snprintf(label_buf, sizeof(label_buf), "ID %d  IN ZONE", p.track_id);
        else
            snprintf(label_buf, sizeof(label_buf), "ID %d  CLEAR", p.track_id);

        int baseline = 0;
        cv::Size tsz = cv::getTextSize(label_buf, cv::FONT_HERSHEY_SIMPLEX, label_scale, label_thick, &baseline);
        int label_y = std::max(tsz.height + baseline + 8, iy1);

        cv::rectangle(frame, cv::Point(ix1, label_y - tsz.height - baseline - 8),
                      cv::Point(ix1 + tsz.width + 12, label_y), color, -1);
        cv::putText(frame, label_buf, cv::Point(ix1 + 6, label_y - baseline - 3),
                    cv::FONT_HERSHEY_SIMPLEX, label_scale, cv::Scalar(0, 0, 0), label_thick, cv::LINE_AA);
    }

    // Banner top-left
    {
        char banner_buf[256];
        snprintf(banner_buf, sizeof(banner_buf),
                 "PERSONS: %d   IN ZONE: %d   ENTRIES: %d",
                 person_count, inside_count, ps.entry_total);

        const double banner_scale = std::max(0.8, 1.1 * (double)scale);
        const int banner_thick = std::max(2, (int)(3.0f * scale));
        int baseline = 0;
        cv::Size tsz = cv::getTextSize(banner_buf, cv::FONT_HERSHEY_SIMPLEX, banner_scale, banner_thick, &baseline);
        int pad = (int)(10.0f * scale) + 6;

        cv::rectangle(frame, cv::Point(0, 0),
                      cv::Point(std::min(W, tsz.width + pad * 2), tsz.height + baseline + pad * 2),
                      cv::Scalar(0, 0, 0), -1);

        cv::Scalar banner_color = (inside_count > 0) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
        cv::putText(frame, banner_buf, cv::Point(pad, tsz.height + pad),
                    cv::FONT_HERSHEY_SIMPLEX, banner_scale, banner_color, banner_thick, cv::LINE_AA);
    }

    write_nv12(frame, in);

    // ── PUBLISH ─────────────────────────────────────────────────────────────
    render_meta::Builder sb(W, H);

    // Publish zone polygon
    {
        std::vector<double> ring;
        for (int i = 0; i < NPOLY; ++i) {
            ring.push_back((double)poly_i[i].x);
            ring.push_back((double)poly_i[i].y);
        }
        sb.polygon(ring, "hazard_zone_1");
    }

    // Publish each tracked person as a box with inside/clear status
    for (size_t pi = 0; pi < persons.size(); ++pi) {
        const PersonTrack &p = persons[pi];
        double bx = (double)p.x1;
        double by = (double)p.y1;
        double bw = (double)(p.x2 - p.x1);
        double bh = (double)(p.y2 - p.y1);
        const char *lbl = p.inside ? "person_IN_ZONE" : "person_CLEAR";
        sb.box(bx, by, bw, bh, lbl, (int)p.track_id, (double)p.score);

        // Publish trail as polyline
        auto tit = ps.tracks.find(p.track_id);
        if (tit != ps.tracks.end() && !tit->second.trail.empty()) {
            std::vector<double> trail_xy;
            for (size_t ti = 0; ti < tit->second.trail.size(); ++ti) {
                trail_xy.push_back((double)tit->second.trail[ti].x);
                trail_xy.push_back((double)tit->second.trail[ti].y);
            }
            char trail_label[64];
            snprintf(trail_label, sizeof(trail_label), "trail_%d", p.track_id);
            sb.polyline(trail_xy, trail_label);
        }
    }

    // Publish scalar fields for downstream sinks
    sb.field_i("person_count", person_count);
    sb.field_i("inside_count", inside_count);
    sb.field_i("entry_total", ps.entry_total);

    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);
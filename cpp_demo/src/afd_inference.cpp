#include "afd_inference.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace afd {
namespace {

// 数值稳定的 sigmoid（避免 x 很负时 exp 溢出）
inline float sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

}  // namespace

SpotDetector::SpotDetector(const std::string& model_path, int num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "afd"),
      options_(),
      in_buf_(static_cast<size_t>(kImgSize) * kImgSize, 0.0f) {

    options_.SetIntraOpNumThreads(num_threads);
    options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ORT 1.31 的路径重载要求 const ORTCHAR_T*（Windows 下为 wchar_t），
    // 这里改用内存加载，跨平台且不受路径编码影响
    std::ifstream mf(model_path, std::ios::binary | std::ios::ate);
    if (!mf) {
        throw std::runtime_error("无法打开模型文件: " + model_path);
    }
    const auto model_size = static_cast<size_t>(mf.tellg());
    model_buf_.resize(model_size);
    mf.seekg(0);
    mf.read(model_buf_.data(), static_cast<std::streamsize>(model_size));
    mf.close();

    session_ = std::make_unique<Ort::Session>(
        env_, model_buf_.data(), model_buf_.size(), options_);

    Ort::AllocatorWithDefaultOptions allocator;
    input_name_  = session_->GetInputNameAllocated(0, allocator).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();

    const auto in_shape =
        session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto out_shape =
        session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    std::cout << "[ORT] input=\"" << input_name_ << "\" shape=[";
    for (auto d : in_shape) std::cout << d << ",";
    std::cout << "]  output=\"" << output_name_ << "\" shape=[";
    for (auto d : out_shape) std::cout << d << ",]";
    std::cout << "\n";
}

Result SpotDetector::infer(const float* data) {
    using clock = std::chrono::high_resolution_clock;
    auto ms_since = [](const clock::time_point& a, const clock::time_point& b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    Result r;
    const size_t plane = static_cast<size_t>(kImgSize) * kImgSize;

    // ---------- 1. 预处理：全局 max 归一化 ----------
    const auto t_pre0 = clock::now();
    float max_val = 0.0f;
    for (size_t i = 0; i < plane; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    const float denom = max_val + 1e-8f;
    for (size_t i = 0; i < plane; ++i) {
        in_buf_[i] = data[i] / denom;
    }
    r.timing.preprocess = ms_since(t_pre0, clock::now());

    // ---------- 2. 推理 ----------
    const auto t_inf0 = clock::now();
    const std::array<int64_t, 4> in_shape{1, 1, kImgSize, kImgSize};

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem_info, in_buf_.data(), in_buf_.size(), in_shape.data(), in_shape.size());

    const char* in_names[]  = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1,
                                 out_names, 1);
    const float* logits = outputs[0].GetTensorData<float>();
    r.timing.inference = ms_since(t_inf0, clock::now());

    // ---------- 3. ROI 搜索 + 通道重排 ----------
    const auto t_roi0 = clock::now();
    r.lut.assign(plane, 0);
    r.min_conf = 1.0f;

    const float module = kImgSize / 3.0f;

    for (int ch = 0; ch < kNumChannels; ++ch) {
        const int m = ch / 4;   // 大模块 0..8
        const int i = m / 3;    // 模块行 0..2
        const int j = m % 3;    // 模块列 0..2
        const int k = ch % 4;   // 0 左上, 1 右上, 2 左下, 3 右下

        const float cx = (j + 0.5f) * module;
        const float cy = (i + 0.5f) * module;
        const int   x0 = std::max(0, static_cast<int>(cx - kSearchRadius));
        const int   x1 = std::min(kImgSize, static_cast<int>(cx + kSearchRadius));
        const int   y0 = std::max(0, static_cast<int>(cy - kSearchRadius));
        const int   y1 = std::min(kImgSize, static_cast<int>(cy + kSearchRadius));

        const float* ch_plane = logits + static_cast<size_t>(ch) * plane;

        float best   = -1e30f;
        int   best_x = x0;
        int   best_y = y0;
        for (int y = y0; y < y1; ++y) {
            const float* row = ch_plane + static_cast<size_t>(y) * kImgSize;
            for (int x = x0; x < x1; ++x) {
                if (row[x] > best) {   // 严格大于，与 np.argmax 的"取第一个"一致
                    best   = row[x];
                    best_x = x;
                    best_y = y;
                }
            }
        }

        const float conf = sigmoid(best);   // 只对 ROI 最大值做 sigmoid
        r.min_conf = std::min(r.min_conf, conf);

        // 通道索引 -> LUT 编号（1..36），完全不依赖坐标排序
        const int row    = i * 2 + (k >= 2 ? 1 : 0);
        const int col    = j * 2 + (k % 2 == 1 ? 1 : 0);
        const int lut_id = col * 6 + (6 - row);

        r.points[lut_id - 1] =
            Point{static_cast<float>(best_x), static_cast<float>(best_y), conf};
    }

    r.timing.roi = ms_since(t_roi0, clock::now());

    // ---------- 4. 置信度预警 ----------
    for (int idx = 0; idx < kNumChannels; ++idx) {
        if (r.points[idx].conf < kConfWarnThresh) {
            std::cout << "警告: 第 " << (idx + 1) << " 通道置信度极低 ("
                      << r.points[idx].conf << ")，可能为暗斑或粘连\n";
        }
    }

    // ---------- 5. 暴力 Voronoi（替代 KDTree） ----------
    const auto t_vor0 = clock::now();
    for (int y = 0; y < kImgSize; ++y) {
        const float fy = static_cast<float>(y);
        for (int x = 0; x < kImgSize; ++x) {
            const float fx = static_cast<float>(x);
            float best_d   = 1e30f;
            int   best_idx = 0;
            for (int p = 0; p < kNumChannels; ++p) {
                const float dx = fx - r.points[p].x;
                const float dy = fy - r.points[p].y;
                const float d  = dx * dx + dy * dy;
                if (d < best_d) {
                    best_d   = d;
                    best_idx = p;
                }
            }
            r.lut[static_cast<size_t>(y) * kImgSize + x] =
                static_cast<uint8_t>(best_idx + 1);
        }
    }
    r.timing.voronoi = ms_since(t_vor0, clock::now());

    return r;
}

void SpotDetector::save_csv(const Result& r, const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("无法写入 " + path);
    f << "index,x,y,conf\n";
    f.setf(std::ios::fixed);
    for (int k = 0; k < kNumChannels; ++k) {
        f.precision(1);
        f << (k + 1) << "," << r.points[k].x << "," << r.points[k].y << ",";
        f.precision(6);
        f << r.points[k].conf << "\n";
    }
}

void SpotDetector::save_lut_raw(const Result& r, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("无法写入 " + path);
    f.write(reinterpret_cast<const char*>(r.lut.data()),
            static_cast<std::streamsize>(r.lut.size()));
}

}  // namespace afd

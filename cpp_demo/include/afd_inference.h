#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace afd {

constexpr int   kImgSize        = 512;
constexpr int   kNumChannels    = 36;
constexpr float kSearchRadius   = 85.0f;   // 3x3 模块半宽
constexpr float kConfWarnThresh = 0.80f;

struct Point {
    float x    = 0.0f;
    float y    = 0.0f;
    float conf = 0.0f;
};

struct Timing {
    double preprocess = 0.0;   // ms
    double inference  = 0.0;
    double roi        = 0.0;
    double voronoi    = 0.0;
};

struct Result {
    std::array<Point, kNumChannels> points{};  // 索引 0 对应 LUT 编号 1
    std::vector<uint8_t>            lut;       // kImgSize*kImgSize，值 1..36
    float                           min_conf = 1.0f;
    Timing                          timing;
};

class SpotDetector {
public:
    /// @param model_path  ONNX 模型路径
    /// @param num_threads 推理线程数，0 = 由 ORT 自动决定（推荐，比单线程快 5 倍）
    explicit SpotDetector(const std::string& model_path, int num_threads = 0);

    /// 输入：kImgSize*kImgSize 的原始 float32（未归一化，行优先）
    Result infer(const float* data);

    static void save_csv(const Result& r, const std::string& path);
    static void save_lut_raw(const Result& r, const std::string& path);

private:
    Ort::Env                          env_;
    Ort::SessionOptions               options_;
    std::unique_ptr<Ort::Session>     session_;
    std::vector<char>                 model_buf_;   // 模型字节，Session 构造后仍需保留
    std::string                       input_name_;
    std::string                       output_name_;
    std::vector<float>                in_buf_;
};

}  // namespace afd

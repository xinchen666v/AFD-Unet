#include "afd_inference.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const std::string model_path = (argc > 1) ? argv[1] : "model/afd_unet.onnx";
    const std::string bin_path   = (argc > 2) ? argv[2] : "test_data/floodmap_10_1.bin";
    const std::string out_prefix = (argc > 3) ? argv[3] : "test_data/cpp";
    const int threads = (argc > 4) ? std::atoi(argv[4]) : 0;  // 0 = ORT 自动
    const int repeat  = (argc > 5) ? std::atoi(argv[5]) : 1;

    // ---- 读取 512x512 float32 原始数据 ----
    std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "无法打开输入文件: " << bin_path << "\n";
        return 1;
    }
    const auto bytes = static_cast<size_t>(f.tellg());
    const size_t count = bytes / sizeof(float);
    f.seekg(0);
    std::vector<float> data(count);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));

    const size_t expect = static_cast<size_t>(afd::kImgSize) * afd::kImgSize;
    if (count != expect) {
        std::cerr << "尺寸不符: 期望 " << expect << " 个 float, 实际 " << count << "\n";
        return 1;
    }
    std::cout << "输入: " << bin_path << "  (" << count << " floats)\n";

    try {
        const auto t_load0 = std::chrono::high_resolution_clock::now();
        afd::SpotDetector detector(model_path, threads);
        const auto t_load1 = std::chrono::high_resolution_clock::now();
        const double load_ms =
            std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
        std::cout << "[ORT] 模型加载: " << load_ms << " ms\n";

        detector.infer(data.data());  // warmup

        std::vector<double> totals;
        afd::Result r;
        for (int it = 0; it < repeat; ++it) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            r = detector.infer(data.data());
            const auto t1 = std::chrono::high_resolution_clock::now();
            totals.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::sort(totals.begin(), totals.end());
        const double median = totals[totals.size() / 2];

        afd::SpotDetector::save_csv(r, out_prefix + "_points.csv");
        afd::SpotDetector::save_lut_raw(r, out_prefix + "_lut.raw");

        std::cout << "\n前 5 个点 (按 LUT 编号):\n";
        for (int k = 0; k < 5; ++k) {
            std::cout << "  #" << (k + 1) << ": (" << r.points[k].x << ", "
                      << r.points[k].y << ")  conf=" << r.points[k].conf << "\n";
        }
        std::cout << "LUT 前 8 像素: ";
        for (int i = 0; i < 8; ++i) std::cout << static_cast<int>(r.lut[i]) << " ";
        std::cout << "\nmin_conf = " << r.min_conf << "\n";

        std::cout << "\n=== 分阶段耗时 (ms) ===\n";
        std::cout << "  预处理 : " << r.timing.preprocess << "\n";
        std::cout << "  推理   : " << r.timing.inference << "\n";
        std::cout << "  ROI    : " << r.timing.roi << "\n";
        std::cout << "  Voronoi: " << r.timing.voronoi << "\n";
        std::cout << "  ---------------------------\n";
        std::cout << "  合计   : "
                  << (r.timing.preprocess + r.timing.inference + r.timing.roi +
                      r.timing.voronoi)
                  << "\n";
        std::cout << "  端到端中位数 (" << repeat << " 次): " << median << " ms\n";
        std::cout << "  线程数 : "
                  << (threads == 0 ? std::string("自动") : std::to_string(threads))
                  << "\n";
        std::cout << "已保存: " << out_prefix << "_points.csv / _lut.raw\n";
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

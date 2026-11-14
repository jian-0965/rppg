# 验证记录

日期：2026-11-14。版本：0.1.0。

## 已执行

- `python -m pip install -e '.[dev]'`：安装成功；发现 OpenCV 5 不提供当前所用的 `CascadeClassifier`，依赖范围已修正为 `opencv-python>=4.8,<5` 并安装验证。
- `python -m pytest -q`：**33 passed**（16.14 秒）。共享环境中的 anyio pytest 插件有一条弃用警告，不影响结果。
- `python -m compileall -q rppg_liveness`：通过。
- `python -m rppg_liveness --help`：三个子命令正常显示。

测试包含 POS／CHROM 对 48、72、105、150 BPM 合成信号的频率恢复（误差断言小于 3 BPM），恒定颜色、独立随机噪声、时间戳抖动、缺失帧、运动和亮度门槛；还有真实 AVI 编解码的无脸拒绝测试、ROI RGB 顺序及多人拒绝测试。

评估集成测试使用**合成预测**检查 CSV／JSON／图片生成和开发集阈值冻结，故意令测试集表现差以验证代码不会利用测试集重新选阈值。测试不是生物识别性能报告。

## 本次运行环境

| 依赖 | 版本 |
| --- | --- |
| Python | 3.12.14 |
| OpenCV | 4.14.0 |
| NumPy | 2.3.5 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.8 |
| pytest | 9.2.0 |

## 未验证／待采集

- 没有提供真人、照片、手机回放等真实标注视频，**真实 Accuracy、APCER、BPCER、ACER、AUC 均未知**。
- 当前执行环境未连接用户摄像头，未进行实体摄像头窗口或 Windows 实机验证。
- 没有接触式 PPG／ECG 真值，未测真实心率误差。
- Python 3.10 / 3.12 的 GitHub Actions 配置已提供；实际远端运行结果以 Actions 页面为准。

下一步按 README 的采集方案填写 manifest，运行 `python -m rppg_liveness evaluate ...`。特别检查回放攻击的 APCER、双方无法判断率和不同拍摄条件的失败分布。

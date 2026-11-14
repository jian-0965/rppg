# rPPG 活体检测实验室（Python）

用普通 RGB 摄像头提取额头、双颊的微弱颜色变化，运行 **POS + CHROM** 脉搏提取与可解释打分，并用独立开发集／测试集评估人脸呈现攻击检测效果。

**当前是可运行的研究基线，没有经过真实真人／攻击数据验证，尚无真实准确率。** 默认分数和阈值不是训练模型，也不是活体概率。高质量视频回放可能保留脉搏信号，不能因为有心率就认定是真人。项目不复现任何论文宣称的准确率。

## 1. 安装与启动

支持 Python 3.10+，普通 CPU，无需 API、登录、GPU或下载额外模型。建议摄像头 640×480 / 30 FPS。首次安装需要联网，运行在本地处理。

```bash
git clone https://github.com/jian-0965/rppg.git
cd rppg
python -m venv .venv
```

Windows PowerShell 不必修改执行策略，可以直接使用虚拟环境内的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m rppg_liveness camera
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m rppg_liveness camera
```

下面命令均假设已激活环境；Windows 也可把 `python` 换成 `.\.venv\Scripts\python.exe`。

**摄像头**：保持单人正脸，均匀稳定光照，等待约 15 秒；后台约每 2 秒发起分析，实际更新延迟取决于 CPU。采集与分析分开运行，避免分析阻塞采样；检测丢失时立即撤销先前结果。按 Q / ESC 结束。界面显示三个采样区域、分数、粗略 BPM 和质量不足原因。默认不保存画面。

```bash
python -m rppg_liveness camera --index 0 --seconds 30 --output outputs/camera.json
```

无图形桌面但有摄像头时：

```bash
python -m rppg_liveness camera --no-display --seconds 30 --output outputs/camera.json
```

**单个视频**：只分析最初 15 秒，不挑选得分最高的一段。视频至少 10 秒，建议录制 20 秒，每次只包含一个对象。

```bash
python -m rppg_liveness video data/live01.mp4 --output outputs/live01.json
```

| 输出 | 意义 |
| --- | --- |
| `live_candidate` | 分数超过阈值，存在较一致的脉搏证据；不保证是真人 |
| `attack_candidate` | 画面质量达到最低要求，但脉搏证据分数低于阈值；不保证是攻击 |
| `inconclusive` | 无法判断，需要重新采集；不应当放行 |
| `score` | 0–1 的启发式证据分数，**不是概率** |
| `bpm` | 42–180 BPM 搜索范围内的估计，仅用于算法观察 |
| `quality` / `reason` | 有效帧比例、帧率、运动、亮度及拒绝原因 |

默认阈值 **0.55 未经验证**，仅用于启动演示。完成评估后使用自己开发集的阈值：

```bash
python -m rppg_liveness camera --calibration outputs/evaluation/calibration.json
python -m rppg_liveness video data/new.mp4 --calibration outputs/evaluation/calibration.json
```

## 2. 怎样测真实准确性

### 采样方案

先做小规模探索实验，再扩大到多人、多设备、多场景。可从 20 位同意参与的人员开始，每人每种条件至少采集 3 次，分不同会话录制；这只是探索规模，不能支撑部署级的低误通过率结论。

| 类别 | 应采集的样本 |
| --- | --- |
| 真人 `live` | 稳定室内光、弱光、自然光；不同肤色、眼镜、轻微动作 |
| 照片 `attack/print` | 彩色打印照片，静止与轻微晃动，光泽／普通纸 |
| 回放 `attack/replay` | 手机／电脑播放真实人脸录像，多种屏幕亮度、距离和刷新率 |
| 屏幕静图 `attack/screen_photo` | 手机／电脑显示静态人脸照片 |
| 可选面具 `attack/mask` | 有条件时单独采集，不能用照片结果代替面具结果 |

每条建议 20 秒，30 FPS，避免美颜、滤镜、自动颜色增强。真人、照片、回放尽量使用**相同摄像头、背景和光照**，否则可能只是测到了设备／场景差异。回放需用摄像头重新拍摄播放屏幕；把原始真人录像直接标成攻击文件并不能测试屏幕呈现攻击。

### 准备 CSV

复制 `examples/manifest.example.csv`，将示例路径替换为自己的视频。示例里的视频没有附带，不能直接拿空文件运行准确率测试。路径相对于 **CSV 所在目录**。

```csv
path,label,subject_id,source_id,split,attack_type
../data/dev/p01_live.mp4,live,p01,p01_session1,dev,none
../data/dev/p02_replay.mp4,attack,p02,p02_original1,dev,replay
../data/test/p03_live.mp4,live,p03,p03_session1,test,none
../data/test/p04_replay.mp4,attack,p04,p04_original1,test,replay
```

- `label` 只能为 `live` 或 `attack`。
- `subject_id` 是画面中被呈现者的身份编号；照片／回放也用原人物编号，不是持屏幕者编号。
- `source_id` 是原始素材或采集会话编号。来自同一原视频的切片、转码、回放及衍生素材应共享该编号。
- `split` 为 `dev` 或 `test`。同一个人及同一原始素材不得跨两组。两组都必须有真人与攻击。
- `attack_type`：真人必须为 `none`，攻击填写 `print`、`replay` 等。

建议按人员约 40% / 60% 分开发集和测试集，保留各种攻击类型。程序检查跨集合身份／素材编号，并拒绝重复路径和内容完全相同的视频。它不能自动识别转码后的同源视频，必须正确标注来源。

### 运行评估

```bash
python -m rppg_liveness evaluate examples/my_manifest.csv --output outputs/evaluation
```

顺序为：提取每段视频特征 → 只在开发集最小化 ACER 选阈值（并列时偏向更少攻击通过）→ 固定阈值评估测试集。没有训练集，因为当前评分权重固定，没有拟合分类器。

如果开发集没有两类均可评分的样本，会保存 `predictions.json` 并明确报错，不编造阈值。测试集结果差，也不能调整阈值后继续把这组数据叫作未见测试集；修改算法后应使用新的保留测试集。

| 文件 | 内容 |
| --- | --- |
| `report.json` | 开发集／测试集指标、按攻击类型结果、覆盖率、区间估计 |
| `predictions.csv` | 每个视频的标签、得分、决定状态及失败原因 |
| `predictions.json` | 逐视频详细特征、质量信息、SHA-256 和时间戳模式 |
| `calibration.json` | 开发集确定的阈值、处理配置与版本 |
| `evaluation.png` | 测试集混淆矩阵和 ROC |

### 优先看什么

| 指标 | 定义／解释 |
| --- | --- |
| APCER | 攻击被错误放行比例，越低越好；尤其关注 `replay` 子组 |
| BPCER | 真人未被放行比例，越低越好 |
| ACER | (APCER + BPCER) / 2，避免类别数量不均衡掩盖问题 |
| Accuracy | 判定正确比例；不能单独代表安全性 |
| Coverage | 能得到有效分数的样本比例，越高越好 |
| AUC | 区分两类的排序能力；本项目只对可评分样本计算 |

主指标明确标为 `*_fail_closed`：质量不足不放行，所以质量不足的真人计入未通过，质量不足的攻击计入拒绝。它们是**本项目的操作策略指标，不是宣称符合 ISO 的认证结果**。报告同时给出双方无法判断率和 `test_scorable_only`，避免靠大量拒绝掩盖识别能力；ROC / AUC 排除质量不足，必须连同覆盖率一起看。混淆矩阵以真人为正类。

报告提供视频级 Wilson 95% 区间。重复人员／同会话样本不独立时，区间会过于乐观；正式研究应增加独立人员并采用人员级 bootstrap 或跨人员交叉验证。即使几条攻击样本全部被拒绝，也不能认定真实误通过率为 0%。

## 3. 算法与限制

1. OpenCV 4.x 自带 Haar 检测器检测单张正脸，缩小到最大宽 480 加速；无需下载模型。
2. 按人脸框比例取额头和双颊，平滑采样框；只剔除接近全黑／饱和像素，没有基于固定肤色的筛选。
3. 读取实际时间戳；短暂缺帧插值，超过 0.25 秒的缺口拒绝。视频后端时间戳不可靠时整段回退到帧号/FPS，并标注 `fps_fallback_assumes_CFR`，可变帧率视频此时应先转为恒定帧率。
4. 默认至少 10 秒、有效采样率 15 FPS、90% 有效帧；检查运动和亮度。摄像头用采集时钟，离线视频不会使用解码耗时估计心率。
5. POS 与 CHROM 投影，滑窗重叠相加，0.7–3.0 Hz 带通、Welch 频谱；丢弃两端各 1 秒的滤波边缘。
6. 比较三处的主频、绝对相关性、双算法一致性和前后半段稳定性。

评分为 `频谱集中度 × (0.30 × 区域频率一致性 + 0.25 × 区域相关性 + 0.25 × 算法一致性 + 0.20 × 时间稳定性)`。这些是人工设定权重，**未经真实数据优化**。频谱集中度只统计分析带内主峰 ±0.15 Hz；`snr_db` 是据此计算的峰带／其余带能量比，不是带接触式真值的临床 SNR。

该实现是 POS／CHROM 思路的独立工程基线，使用密集重叠窗口；并非原论文完整复现或官方 rPPG-Toolbox 代码。Haar 框与固定比例 ROI 在遮挡、侧脸、胡须、头发、暗光和肤色／设备变化下可能偏离皮肤。曝光闪烁、周期运动也可能产生伪脉搏；质量门槛不能保证消除全部伪影。没有身份追踪、主动随机挑战、深度传感或回放专用模型。

如果回放攻击误通过率高，说明仅靠这条信号不足以完成任务。下一步应在独立数据上比较 rPPG + 屏幕／纹理检测、随机动作挑战或深度信息的融合效果。当前 BPM 不作医疗用途，也不含接触式心率真值验证。

## 4. 验证与故障排查

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

测试覆盖已知频率恢复、平坦信号、随机噪声、短片、掉帧、亮度、运动、多人拒绝、RGB 顺序、无脸视频实际解码、开发／测试泄漏、阈值选择、失败样本统计以及报告输出。合成信号／测试通过**不等于真实活体准确率**。仓库 CI 运行 Python 3.10 / 3.12。

| 现象 | 处理 |
| --- | --- |
| 摄像头打不开 | 检查系统权限、关闭占用摄像头的软件，尝试 `--index 1` |
| `face_missing_or_multiple` | 保持单人正脸、脸宽至少约 80 像素；改善光照 |
| `low_fps_or_detection_gap` | 使用 30 FPS 采集、改善光照、减少占用 CPU 的程序 |
| `excessive_motion` | 保持头部稳定；评估移动场景时保留这类失败记录 |
| `unstable_or_extreme_lighting` | 避免逆光、屏幕闪光和自动曝光快速变化 |
| `too_short` | 录制 20 秒并重新运行 |
| `analysis_too_slow` | 摄像头后台分析超过 5 秒，减少 CPU 占用或改用离线视频模式 |
| `no_measurable_color_variation` | 静态或量化后信号太弱，无法确定活体 |
| 缺少 `CascadeClassifier` | 使用本项目约束的 OpenCV 4.x；不支持 OpenCV 5.x 的模块变动 |
| GUI 无法启动 | 安装桌面版 `opencv-python`；无桌面时视频模式可直接用，摄像头用 `--no-display --seconds 30` |

测试版本与验证范围见 [VALIDATION.md](VALIDATION.md)。本项目不会发送人脸到服务端；JSON/CSV 和原始视频请按自身授权管理。`data/`、`outputs/`、常见视频扩展名已加入 `.gitignore`，避免误把人脸视频提交到公开仓库。

## 5. 参考资料

- Wang et al., *Algorithmic Principles of Remote PPG*, IEEE TBME, 2017：[作者机构页面](https://research.tue.nl/en/publications/algorithmic-principles-of-remote-ppg/)。
- de Haan & Jeanne, *Robust Pulse Rate From Chrominance-Based rPPG*, IEEE TBME, 2013：[作者机构页面](https://research.tue.nl/en/publications/robust-pulse-rate-from-chrominance-based-rppg/)。
- Kim et al., *Face Biometric Spoof Detection Method Using a Remote Photoplethysmography Signal*, Sensors, 2022：[开放论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9024982/)。讨论了视频回放保留 rPPG 信号的限制；其数据和模型结果不能套用到本仓库。
- [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)：后续比较深度脉搏提取方法的参考工程。

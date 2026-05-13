# Relumi Agent 架构探索：从千人千面到千图千策

> 创建时间：2026-04-30 ~ 2026-05-12
> 状态：Exploration / Draft
> 关联文档：[global_photography_master_openclaw_inspires_relumi.md](../2.0/global_photography_master_openclaw_inspires_relumi.md)

---

## 目录

1. [整体架构：Go 主链路 + Python Agent Sidecar](#一整体架构go-主链路--python-agent-sidecar)
2. [三层耗时档位控制成本](#二三层耗时档位控制成本)
3. [LangGraph 图结构与关键节点](#三langgraph-图结构与关键节点)
4. [Go 后端集成方案](#四go-后端集成方案)
5. [质量信号反馈与数据库设计](#五质量信号反馈与数据库设计)
6. [是否需要向量数据库](#六是否需要向量数据库)
7. [CLIP vs 多模态 LLM：定位与分工](#七clip-vs-多模态-llm定位与分工)
8. [图片识别是否是"完美提示词"的必须步骤](#八图片识别是否是完美提示词的必须步骤)
9. [从千人千面到千图千策：Image Diagnosis Agent](#九从千人千面到千图千策image-diagnosis-agent)
10. [分层分诊架构：CV + CLIP + LLM](#十分层分诊架构cv--clip--llm)
11. [哲学判断与业界最佳实践](#十一哲学判断与业界最佳实践)
12. [落地路线图](#十二落地路线图)

---

## 一、整体架构：Go 主链路 + Python Agent Sidecar

核心思路：**不改变现有 Go 生产链路的稳定性**，通过 Python Sidecar 提供"智能层"，Go 后端按需调用。

```
┌────────────────── 用户请求 ──────────────────────────────────────────┐
│                                                                       │
│  App → Go Backend (Kratos) → Task Processor → WSC API → 返回结果     │
│           │                       ↑                                   │
│           │ (异步/可选)            │ (增强后的 prompt)                  │
│           ▼                       │                                   │
│  ┌─── Agent Sidecar (Python + LangGraph) ───┐                        │
│  │                                           │                        │
│  │  ┌──────┐   ┌──────────┐   ┌──────────┐ │                        │
│  │  │Vision│ → │Preference│ → │  Prompt  │──┘                        │
│  │  │Analyze│   │  Memory  │   │ Optimize │                          │
│  │  └──────┘   └──────────┘   └──────────┘                          │
│  │       ↓                                                           │
│  │  user_memory (Redis/MySQL)                                        │
│  └───────────────────────────────────────────┘                        │
└───────────────────────────────────────────────────────────────────────┘
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Agent 语言 | Python (不是 Go) | LangGraph 是 Python 生态，兴球 API 用 openai SDK |
| 部署方式 | Sidecar (同机 HTTP) | 延迟可控 (<5ms 网络)，独立伸缩，不影响 Go 主进程 |
| 调用模式 | 同步 + 超时降级 | 500ms 超时，失败用原始 prompt，不阻塞用户 |
| 记忆存储 | Redis (热) + MySQL (冷) | 画像读取 O(1)，信号写入异步 |
| LLM 调用频率 | 仅 10% 请求 | 90% 靠缓存偏好规则，避免多轮 Agent 的成本灾难 |
| 偏好衰减 | halfLife=21d (风格) | 与 OpenClaw 同构，但针对审美场景缩短 |

---

## 二、三层耗时档位控制成本

| 层级 | 触发条件 | 耗时增加 | LLM 调用 | 适用比例 |
|------|---------|---------|----------|---------|
| **L0: 直接透传** | 已有偏好缓存命中 | +5ms (Redis 读) | 0 | ~70% |
| **L1: 轻量增强** | 有用户画像但无缓存 | +200ms | 0 (规则引擎) | ~20% |
| **L2: Agent 分析** | 新用户/偏好漂移/主动触发 | +2~5s | 1-2 次兴球 API | ~10% |

### 成本对比

| 方案 | Token/次 | 月成本 (31K 任务) | 延迟增加/次 |
|------|---------|-----------------|-----------|
| 当前（无 Agent） | 0 | $0 | 0 |
| L0 缓存命中 (70%) | 0 | $0 | +5ms |
| L1 规则增强 (20%) | 0 | $0 | +200ms |
| L2 Agent 分析 (10%) | ~500 | ~$0.24 | +3s |
| **加权平均** | **~50** | **~$0.24** | **+310ms** |
| 文档中 Agent 多轮方案 (已否决) | ~10,000 | ~$47 | +200s |

---

## 三、LangGraph 图结构与关键节点

### 目录结构

```
scripts/agent_sidecar/
├── main.py              # FastAPI 入口
├── graph.py             # LangGraph 图定义
├── nodes/
│   ├── vision.py        # 图片分析节点
│   ├── preference.py    # 偏好查询/更新节点
│   └── prompt.py        # prompt 优化节点
├── memory/
│   ├── store.py         # Redis + MySQL 存储
│   └── decay.py         # 时间衰减算法
├── requirements.txt
└── Dockerfile
```

### LangGraph 状态图

```python
class AgentState(TypedDict):
    # 输入
    user_key: str              # wsid 或 guest_id
    image_url: str             # OSS 图片路径
    function_type: str         # restore/sentence_edit_v2/...
    base_prompt: str           # 当前模板 prompt
    country: str               # X-Country header

    # Agent 中间状态
    image_analysis: Optional[dict]    # 图片分析结果
    user_preference: Optional[dict]   # 用户偏好

    # 输出
    optimized_prompt: str      # 优化后的 prompt
    prompt_suffix: str         # 追加的个性化后缀
    confidence: float          # 优化置信度 (0-1)
    should_cache: bool         # 是否缓存此决策
def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("check_cache", check_preference_cache)
    graph.add_node("analyze_image", analyze_image_with_xinqiu)
    graph.add_node("load_preference", load_user_preference)
    graph.add_node("optimize_prompt", optimize_prompt)
    graph.add_node("update_memory", update_user_memory)

    graph.set_entry_point("check_cache")

    # 条件路由 = 成本控制核心
    graph.add_conditional_edges("check_cache", route_by_cache, {
        "cache_hit": END,                  # L0: 直接返回缓存结果
        "has_profile": "optimize_prompt",  # L1: 用规则优化
        "need_analysis": "analyze_image",  # L2: 完整分析
    })

    graph.add_edge("analyze_image", "load_preference")
    graph.add_edge("load_preference", "optimize_prompt")
    graph.add_edge("optimize_prompt", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()
```

### 关键节点：兴球 API 图片分析

```python
async def analyze_image_with_xinqiu(state: AgentState) -> AgentState:
    """L2 才走这步：分析图片内容、质量、风格"""
    client = OpenAI(base_url="https://new-api.300624.cn/v1", api_key=os.environ["NEW_API_TOKEN"])

    # 精简 prompt，控制 token 消耗 (~300 tokens input + ~200 output)
    response = client.chat.completions.create(
        model="gpt-5.4(high)",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": state["image_url"]}},
                {"type": "text", "text": (
                    "Analyze this photo in JSON format: "
                    '{"quality": "high/medium/low", '
                    '"style": "warm/cool/neutral/vibrant", '
                    '"subject": "portrait/landscape/object/group", '
                    '"lighting": "natural/studio/golden_hour/low_light", '
                    '"suggested_enhancement": "brief description"}'
                )}
            ]
        }],
        max_tokens=200,
        stream=True,
    )

    result = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            result += chunk.choices[0].delta.content

    state["image_analysis"] = json.loads(result)
    return state
```

### 关键节点：偏好记忆 + 时间衰减

```python
def calculate_decay(age_days: float, half_life: float = 21.0) -> float:
    """OpenClaw 同构时间衰减"""
    lam = math.log(2) / half_life
    return math.exp(-lam * max(0, age_days))
```

### 关键节点：Prompt 优化（规则 + 少量 LLM）

```python
# 地区审美映射（Apollo 热配置，这里是默认值）
REGION_AESTHETIC_MAP = {
    "south_asia": "Bright, even illumination. Warm color tones. Vibrant and flattering.",
    "east_asia_jp": "Soft, low contrast. Muted elegant tones. Airy feel.",
    "east_asia_cn": "Smooth skin, bright, creamy tones.",
    "middle_east": "Golden warm tones, high contrast, luxurious feel.",
    "latin_america": "Warm tones, high saturation, vibrant.",
    "western": "",  # 默认风格，不追加
}
```

---

## 四、Go 后端集成方案

在现有 `processor` 中加入 Agent 调用（可选/异步），超时 500ms，失败 fallback 到原始 prompt：

```go
// internal/biz/processor/agent_enhance.go

type AgentEnhanceRequest struct {
    UserKey      string `json:"user_key"`
    ImageURL     string `json:"image_url"`
    FunctionType string `json:"function_type"`
    BasePrompt   string `json:"base_prompt"`
    Country      string `json:"country"`
}

type AgentEnhanceResponse struct {
    OptimizedPrompt string  `json:"optimized_prompt"`
    PromptSuffix    string  `json:"prompt_suffix"`
    Confidence      float64 `json:"confidence"`
}

func (p *FunctionProcessors) EnhancePromptWithAgent(ctx context.Context, req *AgentEnhanceRequest) (string, error) {
    ctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
    defer cancel()

    // POST http://localhost:8100/enhance
    // Agent 不可用时静默降级，返回 req.BasePrompt
}
```

---

## 五、质量信号反馈与数据库设计

```sql
-- 质量信号表
CREATE TABLE user_quality_signals (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_key VARCHAR(64) NOT NULL,
    task_id VARCHAR(64) NOT NULL,
    function_type VARCHAR(32) NOT NULL,
    signal_type ENUM('download', 'share', 'delete', 're_edit', 'rate_good', 'rate_bad') NOT NULL,
    signal_value INT NOT NULL,  -- +3/+5/-1/-2/+4/-3
    prompt_used TEXT,
    prompt_suffix_used VARCHAR(512),
    created_at BIGINT NOT NULL,
    INDEX idx_user_key_time (user_key, created_at)
) ENGINE=InnoDB;

-- 用户偏好聚合表（定时任务写入）
CREATE TABLE user_memory_profiles (
    user_key VARCHAR(64) PRIMARY KEY,
    region VARCHAR(32) NOT NULL DEFAULT 'western',
    top_functions JSON,           -- {"sentence_edit_v2": 0.85, "restore": 0.12}
    style_preferences JSON,      -- {"warm_ratio": 0.73, "high_sat_ratio": 0.61}
    prompt_suffix_cache VARCHAR(512),
    signal_count INT NOT NULL DEFAULT 0,
    last_active_at BIGINT,
    updated_at BIGINT,
    INDEX idx_last_active (last_active_at)
) ENGINE=InnoDB;
```

---

## 六、是否需要向量数据库

### 分阶段判断

| 阶段 | 是否需要 | 理由 |
|------|---------|------|
| Phase 0-1 | **不需要** | 模板 <100 个，结构化特征 + 规则引擎足够 |
| Phase 2 | **灰色地带** | 当用户历史 prompt 积累到 100+ 条，关键词抽取丢失语义 |
| Phase 3 | **大概率需要** | 做"根据图片内容自动匹配最佳 prompt"时，这是 image→text 检索问题 |

### 三类需要向量化的对象

```
① 用户的 prompt（自然语言输入）
   → 维度: 文本嵌入
   → 匹配目标: 找到相似审美偏好

② 用户的图片（视觉特征）
   → 维度: CLIP 嵌入
   → 匹配目标: 找到相似构图/风格的历史成功案例

③ 历史成功 prompt 片段
   → 维度: 文本嵌入
   → 匹配目标: 找到最接近用户意图的已验证 prompt
```

### 推荐方案：Redis Vector Search

```
方案 1: pgvector (PostgreSQL) — 需引入新组件
方案 2: Redis Vector Search  — ✅ 最契合，已有 Redis，零新增组件
方案 3: SQLite + sqlite-vec  — 适合 Agent Sidecar 内嵌

推荐方案 2：
  FT.CREATE idx ON HASH PREFIX 1 prompt:
    SCHEMA embedding VECTOR FLAT 6 DIM 768 ...
  FT.SEARCH idx "*=>[KNN 5 @embedding $vec]"
```

### 向量检索 vs LLM 调用的取舍

| 维度 | 向量检索 | LLM Vision |
|------|---------|------------|
| 每次成本 | ~0 token | ~500 token |
| 每次延迟 | ~10ms (Redis) + 50ms (CLIP) | ~3s (兴球 API) |
| 冷启动 | 需要积累历史数据 | 首次即可用 |
| 精度来源 | 用户自身的历史行为 | 模型的通用理解 |
| 可积累性 | ✅ 越多数据越准 | ❌ 每次独立 |

**最优组合**：新用户用 LLM（冷启动兜底），老用户用向量检索（从自身历史学习）。

---

## 七、CLIP vs 多模态 LLM：定位与分工

### 本质区别

| 维度 | CLIP | 多模态 LLM (GPT-5.4 / Gemini) |
|------|------|------|
| 本质 | 编码器 (Encoder) | 生成器 (Generator) |
| 输出 | 768 维浮点数向量 | 自然语言文本 |
| 类比 | 给照片贴"指纹"，用来比对 | 请评论家写一段影评 |
| 速度 | ~10-50ms (GPU) / ~100ms (CPU) | ~2-5 秒 (API) |
| 成本 | 自部署后趋近于 0 | ~500 token/次 |
| 能做 | 判断相似度、聚类、风格匹配 | 描述内容、回答问题、生成 prompt |
| 不能做 | 告诉你图片里有什么 | 高效比较两张图的相似度 |

**一句话总结**：CLIP 是"比较器"，LLM 是"理解器"。CLIP 回答"A 和 B 像不像"，LLM 回答"A 是什么"。

### CLIP 在 Relumi 中的真正价值

不是"理解图片里有什么"，而是：

> **在用户自身历史中检索——"你上次处理类似照片时，满意的 prompt 是什么？"**

```
用户上传新图片
  → CLIP 嵌入 → 768 维向量
  → Redis Vector Search: 在该用户历史中找最相似的图片
  → 取其 quality_signal 最高的 prompt
  → 作为本次 prompt 基础/参考
  → 零 LLM 调用，~60ms
```

---

## 八、图片识别是否是"完美提示词"的必须步骤

### 提示词的三层结构

```
Layer 1: 任务意图 (What to do)
  "Transform the lighting to golden hour"
  来源：用户选择的模板/功能 → 已知，不需要识别

Layer 2: 图片上下文 (What's in the photo)
  "There is a person standing on a beach"
  来源：图片内容 → 端到端多模态模型自己能看到图片
  → 大部分场景下是冗余的（用一个 LLM 描述图片给另一个 LLM 看）

Layer 3: 审美偏好 (How it should feel)
  "Rich warm golden palette. High saturation. Cinematic."
  来源：用户个人偏好 → 记忆系统的核心价值
```

### 结论

| 目标 | 是否必须识别图片 |
|------|----------------|
| 让 prompt 更符合用户审美 | 不一定 |
| Scene Retake 换光照 | 不一定 |
| 老照片修复，保持人脸特征 | 建议 |
| 手机翻拍实体照片，需要裁切/透视矫正 | **必须** |
| 判断黑白照片是否上色 | **必须** |
| 自动选择 restore / sharpen / colorize 链路 | **必须** |

### 业界三大流派

```
流派 1: Midjourney / DALL-E
  不分析输入图片。prompt 就是一切。
  哲学：把创造力还给用户，模型是工具。

流派 2: Google Photos / Apple Photos
  深度分析每张图片，自动增强。用户几乎不需要输入。
  哲学：AI 全自动搞定。千人一面。

流派 3: Adobe Lightroom AI / Luminar  ← Relumi 的目标位置
  分析图片内容 + 学习用户的编辑历史。
  "你通常会把人像照片调暖 +15，加对比度 +10"
  哲学：AI 学习你的风格，成为你的助手。
```

---

## 九、从千人千面到千图千策：Image Diagnosis Agent

### 需求升级

```
千人千面 (User-aware)：这个用户喜欢什么风格？
千图千策 (Image-aware)：这张图本身需要什么处理？

真正的个人摄影大师 = User-aware + Image-aware + Task-aware

最终处理策略 =
  用户意图
+ 图片诊断结果
+ 用户历史偏好
+ 当前功能能力
+ 成本/耗时约束
```

### 识别目标不是"内容描述"，而是"处理策略诊断"

不推荐：

> 这是一张老照片，里面有一个老人，背景是室内，照片边缘有钱包皮夹子……

推荐——结构化诊断：

```json
{
  "image_source_type": "phone_capture_of_physical_photo",
  "photo_boundary_detected": true,
  "needs_crop": true,
  "crop_confidence": 0.92,
  "is_black_white": true,
  "face_count": 1,
  "face_quality": "low",
  "face_preservation_priority": "high",
  "damage_types": ["scratch", "stain", "fade", "blur"],
  "reflection_detected": true,
  "perspective_distortion": "medium",
  "recommended_pipeline": [
    "detect_photo_boundary",
    "perspective_correct",
    "crop_to_photo",
    "reflection_reduce",
    "restore_damage",
    "face_preserve_enhance",
    "optional_colorize"
  ]
}
```

### 钱包老照片典型案例

```
用户上传：手机拍摄的、嵌在钱包/皮夹子里的破损老照片

如果直接丢给 restore 模型：
  - 可能把钱包也当成图像主体修复
  - 保留皮夹子边框
  - 被透视角度干扰
  - 人脸区域像素太小，身份保持差

合理流程：
  Step 1: 检测"照片区域"边界
  Step 2: 判断是否需要透视矫正
  Step 3: 裁切出真正老照片
  Step 4: 判断是否有反光/阴影/遮挡
  Step 5: 判断是否黑白/褪色/破损/模糊
  Step 6: 选择 restore prompt 和算法参数
  Step 7: 人脸区域单独增强或 identity-preserving 修复
  Step 8: 可选上色
```

### 黑白照片上色策略

| 场景 | 建议 |
|------|------|
| 用户只选择"restore/修复" | 默认保持黑白，修复优先 |
| 用户明确选择"colorize" | 上色 |
| App 提供"修复并上色"选项 | 用户授权 |
| 画像显示用户偏好彩色化 | 个性化依据 |
| 黑白照片修复完成后 | 推荐："要不要尝试彩色化版本？" |

---

## 十、分层分诊架构：CV + CLIP + LLM

### Image Diagnosis Layer

```
┌──────────────────────────────────────────────────────────────────┐
│                         Image Diagnosis Layer                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  L0: 传统 CV / 轻量模型            成本低，速度快，确定性强       │
│      - 分辨率、模糊、曝光、噪声                                   │
│      - 黑白/彩色判断                                              │
│      - 边缘/矩形边界检测                                          │
│      - 人脸检测 (dlib / mediapipe)                                │
│      - EXIF / 图像尺寸                                            │
│                                                                  │
│  L1: CLIP / 分类模型               成本中等，适合语义分类         │
│      - printed photo vs digital photo                             │
│      - wallet/frame/table/background context                      │
│      - old photo / document / portrait / landscape                 │
│                                                                  │
│  L2: 多模态 LLM                    成本高，只做兜底和复杂判断     │
│      - L0/L1 置信度低                                             │
│      - 图片复杂（多人、复杂背景）                                 │
│      - 高价值用户/高积分任务                                      │
│      - 需要生成结构化修复策略                                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 三者定位

| 技术 | 最适合做什么 | 不适合做什么 |
|------|------------|------------|
| 传统 CV | 快速检测边框、模糊、亮度、黑白、透视、人脸 | 理解复杂语义 |
| CLIP | 相似图检索、风格匹配、历史案例匹配、零次分类 | 精准定位边界、生成诊断 |
| 多模态 LLM | 复杂图像理解、模糊场景判断、策略解释 | 每张图都调用（成本/延迟高） |

### LLM 调用时机

| 场景 | 是否调用 LLM |
|------|------------|
| 普通清晰人像，用户选择 Scene Retake | 不需要 |
| 普通老照片扫描件 | 不一定 |
| 手机翻拍实体照片，边界不清 | 建议 |
| 钱包/相框/相册页中的照片 | 建议 |
| 多张脸，主体不明确 | 建议 |
| 严重破损且人脸模糊 | 建议 |
| 高价值付费用户 | 可提高调用比例 |
| 前处理判断置信度低 | 必须 |

---

## 十一、哲学判断与业界最佳实践

### 核心原则

> **能用结构化感知解决的问题，不要交给语言；**
> **能用确定性处理解决的问题，不要交给生成模型；**
> **只有审美和意图的不确定性，才交给 Agent。**

### 四层分工

| 层级 | 回答的问题类型 | 技术选择 |
|------|-------------|---------|
| 结构化感知 | 事实问题：是否模糊？是否有边框？是否黑白？ | CV / 轻量模型 / CLIP |
| 确定性前处理 | 工程问题：该不该裁切？该不该透视矫正？ | 传统算法 / 固定 pipeline |
| 生成式模型处理 | 创作问题：修复成什么风格？人脸是否保持身份？ | 大模型 + prompt |
| 记忆系统 | 个性化问题：用户更喜欢什么？是否讨厌过度磨皮？ | 用户行为信号 + 时间衰减 |

### prompt 不是魔法咒语

```
垃圾输入 + 完美 prompt ≠ 完美输出

正确的输入裁切 + 正确的任务拆解 + 正确的模型选择 + 正确的 prompt = 好的输出
```

### 完美提示词的真正定义

```
完美提示词 =
  对图片状态的正确诊断
+ 对用户意图的正确理解
+ 对模型能力边界的正确约束
+ 对历史成功经验的复用
```

### 医疗类比

```
不合理：病人来了 → 直接手术
合理：  病人来了 → 分诊 → 检查 → 诊断 → 制定治疗方案 → 执行 → 复查

Agent 更像"影像分诊医生"，不是每次都亲自动刀。
LLM 不做常规入口，LLM 做疑难会诊。
```

---

## 十二、落地路线图

### Phase 0-1: 基础设施 + 规则引擎（不需要向量库）

```
Week 1: 基础设施
  ├── 创建 user_quality_signals 表
  ├── 任务完成时写入信号（Go 侧，异步写入）
  └── 搭建 Python Agent Sidecar 骨架 (FastAPI + LangGraph)

Week 2: L0 + L1 路径
  ├── 实现 region → prompt_suffix 规则引擎
  ├── Go 后端读取 X-Country 注入 suffix
  ├── Redis 缓存用户偏好
  └── 聚合定时任务（信号 → 画像）

Week 3: L2 Agent 路径
  ├── 集成兴球 API 图片分析节点
  ├── LangGraph 条件路由（新用户/偏好漂移触发）
  └── Go 后端 500ms 超时调用 + 降级逻辑

Week 4: 闭环验证
  ├── A/B 测试：对比有/无 Agent 增强的满意度
  ├── 监控：token 消耗、延迟分布、缓存命中率
  └── 调优：衰减半衰期、信号权重、路由阈值
```

### Phase 2: 向量能力 + 图片诊断

```
  ├── 引入 Redis Vector Search 或 CLIP 嵌入
  ├── 传统 CV 前处理管道（边界检测、透视矫正、黑白判断）
  ├── user_memory_profiles 表 + 定时聚合 + Decay 算法
  ├── 个性化模版排序（GetConfig 读取偏好）
  └── 探索推荐位 (MMR 多样性)
```

### Phase 3: Image Treatment Planner

```
  ├── 完整 Image Diagnosis Agent（CV + CLIP + LLM 分层分诊）
  ├── 处理链路路由（crop → perspective → restore → colorize → face enhance）
  ├── Prompt 个性化增强（基于偏好的 prompt suffix）
  ├── 反光检测 + 去除
  └── 群体审美自动调优（聚合全体信号 → 更新 region 参数）
```

### 最终目标架构：Image Treatment Planner

```
用户上传图片
  ↓
L0: 快速图像体检 (CV)
  分辨率 / 模糊度 / 黑白 / 亮度 / 人脸 / 是否翻拍
  ↓
L1: 结构化诊断 (CLIP + 规则)
  是否需要裁切 / 透视矫正 / 去反光 / 人脸保护 / 上色建议
  ↓
L2: 策略路由
  restore only / crop→restore / crop→perspective→restore / restore→face enhance
  ↓
L3: Prompt + 参数生成
  基础 prompt + 图片诊断 suffix + 人脸保持约束 + 用户偏好 suffix
  ↓
模型执行
  ↓
后处理 / 质检 / 推荐下一步
```

输出的不是一段 prompt，而是一份**处理计划**：

```json
{
  "diagnosis": { ... },
  "plan": [
    {"step": "crop_photo_boundary", "reason": "..."},
    {"step": "perspective_correct", "reason": "..."},
    {"step": "restore", "prompt_suffix": "..."},
    {"step": "colorize_optional", "reason": "..."},
    {"step": "portrait_enhance", "condition": "if face confidence remains low"}
  ],
  "final_prompt": "Restore this old photo naturally. Preserve original facial identity..."
}
```

---

*文档创建：2026-05-12*
*来源：Agent 架构探索对话 (2026-04-30 ~ 2026-05-13)*

---

## 附录：POC 实验记录 (2026-05-12 ~ 2026-05-13)

> 以下实验在 `tmp/agent_demo/` 中进行，使用 WSC Gemini Banana V2 API 作为 LLM 图像编辑后端。

### 实验 1：合照开眼（Group Photo Open Eyes）

**假设**：多人合照场景下，LLM 全图一次调用时注意力分散，导致开眼质量不均。通过 CV 人脸分治可提升效果。

#### 方案对比

| | 对照组 (Baseline) | 实验组 (Agent) |
|---|---|---|
| 流程 | 生产固定 prompt → 全图单次 WSC | CV 人脸检测 → 全图增强 → 逐脸裁切 → 单独 WSC → 融合 |
| Prompt | 生产环境 2971 chars | 全图 prompt + 人脸专用 prompt |
| WSC 调用 | 1 次 | 6 次 (1 全图 + 5 人脸) |
| 耗时 | 35.9s | 221.2s |
| 人脸检测 | 无 | insightface SCRFD 5 张 |

#### 技术栈

- **人脸检测**: insightface 0.7.3 (SCRFD buffalo_sc, CPU)，5 点关键点 + 置信度
- **融合算法**: 高斯模糊椭圆 Alpha Mask（泊松融合因 WSC 返回尺寸不一致会崩溃）
- **裁切策略**: `FACE_PAD_RATIO=1.8`，取人脸 bbox 扩展 1.8x 的正方形区域

#### 关键发现

1. **分辨率增益显著**：3280×4928 全图中每张人脸约 300-500px 宽，裁切放大后给模型约 856-1000px 输入，眼部细节可编辑空间大幅增加
2. **注意力聚焦有效**：单次 WSC 只处理 1 张脸，不存在"顾此失彼"问题
3. **模型返回尺寸不可控**：WSC 固定返回 1024×1024，需 resize 回原裁切尺寸再融合
4. **比例保持是难点**：Face 2 出现过"放大脑袋"问题，需在 prompt 中加强比例约束

#### 结论

> **CV 人脸分治在合照开眼场景价值确认**。核心价值不在"丰富 prompt"（多模态模型自己能看到图），而在**预处理管线**：提升每张人脸的输入分辨率 + 集中模型注意力。

---

### 实验 2：CRT 翻拍照片去网格 + 修复

**假设**：CRT/LCD 翻拍照片存在像素网格纹理，LLM 可能将其误判为图像本身纹理而保留。CV 预处理去除网格后再送 LLM 效果更好。

#### 方案对比

| | 对照组 (Baseline) | 实验组 (Agent) |
|---|---|---|
| 流程 | 原图（含像素网格）直接 → WSC LLM | CV 去像素网格 → WSC LLM |
| CV 预处理 | 无 | `remove_pixel_grid()` 0.09s |
| WSC 调用 | 1 次 | 1 次 |
| LLM 耗时 | 48.7s | 31.2s |
| 总耗时 | 48.7s | 31.3s |

#### 技术演进

最初误判为"摩尔纹"尝试 FFT 陷波滤波 → 9992 个峰值导致性能爆炸（全图高斯×万次）→ 诊断为 CRT 像素网格翻拍 → 改用正确方案：

| 方法 | 适用场景 | 本实验效果 |
|---|---|---|
| FFT 陷波滤波 | 规则摩尔纹（几十个频率峰） | ❌ 不适用（万级峰值） |
| 下采样+中值+双边滤波 | CRT/LCD 像素网格 | ✅ 完美去除 |

#### `remove_pixel_grid()` 算法

```
输入 (3024×3024, 含 RGB 子像素网格)
  ↓
自动检测网格周期（自相关函数找峰值）→ period=4px
  ↓
高斯低通模糊 (ksize=5, 抗混叠)
  ↓
下采样到 ≥800px (INTER_AREA)
  ↓
中值滤波 (k=3, 去残余网点)
  ↓
双边滤波 (d=5, σ_color=40, σ_space=40, 保边平滑)
  ↓
Lanczos 上采样回原始分辨率
  ↓
USM 锐化 (1.3 * img - 0.3 * blur, σ=2.0)
  ↓
输出 (3024×3024, 网格完全消除)
```

#### 关键发现

1. **类型判别很重要**：9992 个频率峰值 = 不是摩尔纹，是密集像素网格。方法选错会卡死
2. **CV 预处理让 LLM 更快**：输入更干净后 LLM 耗时从 48.7s → 31.2s（节省 36%），模型处理简单图像更高效
3. **性能陷阱**：原版 FFT 陷波对每个峰值做全图尺寸数组运算，万级峰值时 O(n×W×H) 爆炸。修复为局部 patch + 峰值上限

#### 结论

> **CV 去网格预处理价值确认**：不仅提升修复质量（去除干扰纹理），还降低 LLM 耗时和成本。属于典型的"清理输入 → 帮助模型聚焦"模式。

---

### 实验总结：Agent 核心价值模型

```
Agent 收益 = f(子问题独立性 × 单问题复杂度 × 模型注意力竞争度)
```

| 已验证场景 | CV 角色 | 价值类型 | 收益评级 |
|---|---|---|---|
| 合照开眼 | 人脸检测+裁切+融合 | 分辨率增益 + 注意力聚焦 | ⭐⭐⭐⭐⭐ |
| CRT 翻拍修复 | 像素网格去除 | 清理输入 + 降低模型负担 | ⭐⭐⭐⭐ |

### CV Tools 清单

| Tool | 实现状态 | 文件 | 用途 |
|---|---|---|---|
| `analyze()` | ✅ 已实现 | `cv/analyzer.py` | 亮度/模糊度/灰度/人脸检测 |
| `detect_moire()` | ✅ 已实现 | `cv/demoire.py` | FFT 频谱摩尔纹检测 |
| `remove_moire()` | ✅ 已实现 | `cv/demoire.py` | FFT 陷波去摩尔纹 |
| `remove_pixel_grid()` | ✅ 已实现 | `cv/demoire.py` | 下采样融合去像素网格 |
| `blend_face_back()` | ✅ 已实现 | `test_openeyes_ab.py` | 椭圆 Alpha 人脸融合 |
| `face_encode` | 📋 待实现 | — | 人脸编码/跨帧一致性校验 |
| `segment_anything` | 📋 待实现 | — | 通用分割（场景重拍抠图） |
| `depth_estimate` | 📋 待实现 | — | 深度估计（透视合成） |

### 待验证场景

| 场景 | Agent 策略 | 预期价值 | 优先级 |
|---|---|---|---|
| 老照片多人修复 | 人脸分治 + 分区上色 + 损伤 mask inpainting | 避免串色/顾此失彼 | P0 |
| 场景重拍光照 | 深度估计 + 光照方向推断 + 人物重光照 | 融合自然度 | P1 |
| 图生视频身份保持 | 关键帧人脸编码对比 + 不一致帧重生成 | 角色一致性 | P1 |
| 自定义生图多元素 | Prompt 分治 + 逐区域精化 + 风格统一 | 复杂需求控制力 | P2 |
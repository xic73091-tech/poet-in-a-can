"""记忆蒸馏算法 - 高级功能测试

覆盖现有 test_memory.py 未测试的部分：
- ForgettingEngine（遗忘引擎）
- MemorySystem（系统集成）
- 双语映射字典
- 边界条件与极端情况
- MemoryFragment 深层属性
"""

import sys
import os
import time
import importlib
from unittest.mock import patch

import pytest

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "02_生命机制"))

memory_module = importlib.import_module("记忆蒸馏算法")

MemoryFragment = memory_module.MemoryFragment
ShortTermMemory = memory_module.ShortTermMemory
LongTermMemory = memory_module.LongTermMemory
DistillationEngine = memory_module.DistillationEngine
ForgettingEngine = memory_module.ForgettingEngine
MemorySystem = memory_module.MemorySystem
EmotionTag = memory_module.EmotionTag
SourceType = memory_module.SourceType
EMOTION_API_MAP = memory_module.EMOTION_API_MAP
SOURCE_API_MAP = memory_module.SOURCE_API_MAP
API_EMOTION_MAP = memory_module.API_EMOTION_MAP
API_SOURCE_MAP = memory_module.API_SOURCE_MAP


# ─────────────────────────────────────────────
# 辅助工厂函数
# ─────────────────────────────────────────────

def make_fragment(
    content="测试",
    source=SourceType.TIME,
    emotion=EmotionTag.CALM,
    intensity=0.5,
    timestamp=None,
    decay_rate=0.1,
    access_count=0,
):
    """快速创建 MemoryFragment，可控制 timestamp"""
    frag = MemoryFragment(
        content=content,
        source=source,
        emotion=emotion,
        intensity=intensity,
        decay_rate=decay_rate,
        access_count=access_count,
    )
    if timestamp is not None:
        frag.timestamp = timestamp
    return frag


# ═════════════════════════════════════════════
# 1. 双语映射字典测试
# ═════════════════════════════════════════════

class TestBilingualMaps:
    """EMOTION_API_MAP / SOURCE_API_MAP / API_EMOTION_MAP / API_SOURCE_MAP"""

    def test_emotion_api_map_has_all_emotions(self):
        """每个 EmotionTag 枚举值都应有对应的 API 映射"""
        for tag in EmotionTag:
            assert tag.value in EMOTION_API_MAP, f"{tag.name} ({tag.value}) 不在 EMOTION_API_MAP 中"

    def test_source_api_map_has_all_sources(self):
        """每个 SourceType 枚举值都应有对应的 API 映射"""
        for tag in SourceType:
            assert tag.value in SOURCE_API_MAP, f"{tag.name} ({tag.value}) 不在 SOURCE_API_MAP 中"

    def test_emotion_map_count_matches_enum(self):
        assert len(EMOTION_API_MAP) == len(EmotionTag)

    def test_source_map_count_matches_enum(self):
        assert len(SOURCE_API_MAP) == len(SourceType)

    def test_api_emotion_map_is_reverse(self):
        """API_EMOTION_MAP 应该是 EMOTION_API_MAP 的反向映射"""
        for api_val, cn_val in API_EMOTION_MAP.items():
            assert EMOTION_API_MAP[cn_val] == api_val

    def test_api_source_map_is_reverse(self):
        """API_SOURCE_MAP 应该是 SOURCE_API_MAP 的反向映射"""
        for api_val, cn_val in API_SOURCE_MAP.items():
            assert SOURCE_API_MAP[cn_val] == api_val

    def test_roundtrip_emotion(self):
        """中文 → API英文 → 中文 应该能还原"""
        for cn, api in EMOTION_API_MAP.items():
            assert API_EMOTION_MAP[api] == cn

    def test_roundtrip_source(self):
        """中文 → API英文 → 中文 应该能还原"""
        for cn, api in SOURCE_API_MAP.items():
            assert API_SOURCE_MAP[api] == cn

    def test_emotion_api_values_are_unique(self):
        values = list(EMOTION_API_MAP.values())
        assert len(values) == len(set(values))

    def test_source_api_values_are_unique(self):
        values = list(SOURCE_API_MAP.values())
        assert len(values) == len(set(values))

    def test_emotion_api_values_are_lowercase_english(self):
        for api_val in EMOTION_API_MAP.values():
            assert api_val.isascii() and api_val.islower(), f"'{api_val}' 不是纯小写英文"

    def test_source_api_values_are_lowercase_english(self):
        for api_val in SOURCE_API_MAP.values():
            assert api_val.isascii() and api_val.islower(), f"'{api_val}' 不是纯小写英文"


# ═════════════════════════════════════════════
# 2. MemoryFragment 深层测试
# ═════════════════════════════════════════════

class TestMemoryFragmentAdvanced:
    """MemoryFragment 边界条件与深层行为"""

    def test_age_is_non_negative(self):
        frag = make_fragment()
        assert frag.age >= 0

    def test_age_reflects_time(self):
        """用伪造的 timestamp 验证 age 计算"""
        now = time.time()
        frag = make_fragment(timestamp=now - 3600)  # 1小时前
        assert frag.age >= 3599  # 允许微小误差
        assert frag.age <= 3601

    def test_importance_with_zero_intensity(self):
        """零强度 → 重要度应为 0"""
        frag = make_fragment(intensity=0.0)
        assert frag.importance == 0.0

    def test_importance_bounded_above(self):
        """重要度不应超过 intensity * 1.5（最大访问加成）"""
        frag = make_fragment(intensity=1.0, decay_rate=0.0)
        for _ in range(100):  # 远超上限
            frag.touch()
        # access_bonus = min(100 * 0.1, 0.5) = 0.5
        # importance = 1.0 * (1 + 0.5) * exp(0) = 1.5
        assert frag.importance <= 1.5 + 1e-9

    def test_importance_decreases_over_time(self):
        """相同 fragment，时间越久重要度越低"""
        now = time.time()
        f_recent = make_fragment(intensity=0.8, timestamp=now - 60)
        f_old = make_fragment(intensity=0.8, timestamp=now - 86400 * 30)  # 30天前
        assert f_recent.importance > f_old.importance

    def test_touch_updates_last_accessed(self):
        frag = make_fragment()
        assert frag.last_accessed == 0
        frag.touch()
        assert frag.last_accessed > 0

    def test_decay_rate_affects_importance(self):
        """更高的衰减率 → 更低的重要度（对于老记忆）"""
        now = time.time() - 86400 * 7  # 7天前
        f_fast = make_fragment(intensity=0.8, decay_rate=0.5, timestamp=now)
        f_slow = make_fragment(intensity=0.8, decay_rate=0.01, timestamp=now)
        assert f_slow.importance > f_fast.importance

    def test_zero_decay_rate_no_time_loss(self):
        """衰减率为 0 时，时间不应影响重要度"""
        now = time.time()
        frag = make_fragment(intensity=0.7, decay_rate=0.0, timestamp=now - 86400 * 365)
        # importance = 0.7 * (1 + 0) * exp(0) = 0.7
        assert abs(frag.importance - 0.7) < 1e-9

    def test_access_bonus_cap(self):
        """访问加成上限为 0.5"""
        frag = make_fragment(intensity=1.0, decay_rate=0.0)
        for _ in range(50):
            frag.touch()
        # 50 * 0.1 = 5.0, min(5.0, 0.5) = 0.5
        # importance = 1.0 * 1.5 = 1.5
        assert abs(frag.importance - 1.5) < 1e-9

    def test_empty_content(self):
        """内容为空字符串应合法"""
        frag = make_fragment(content="")
        assert frag.content == ""

    def test_max_intensity(self):
        """强度为 1.0"""
        frag = make_fragment(intensity=1.0)
        assert frag.intensity == 1.0


# ═════════════════════════════════════════════
# 3. ShortTermMemory 边界测试
# ═════════════════════════════════════════════

class TestShortTermMemoryAdvanced:

    def test_empty_memory_len(self):
        stm = ShortTermMemory()
        assert len(stm) == 0

    def test_get_recent_on_empty(self):
        stm = ShortTermMemory()
        assert stm.get_recent(5) == []

    def test_get_by_emotion_no_match(self):
        stm = ShortTermMemory(capacity=10)
        stm.add(make_fragment(emotion=EmotionTag.JOY))
        result = stm.get_by_emotion(EmotionTag.SADNESS)
        assert result == []

    def test_get_by_emotion_on_empty(self):
        stm = ShortTermMemory()
        assert stm.get_by_emotion(EmotionTag.JOY) == []

    def test_get_recent_more_than_available(self):
        stm = ShortTermMemory(capacity=10)
        stm.add(make_fragment(content="a"))
        stm.add(make_fragment(content="b"))
        result = stm.get_recent(10)
        assert len(result) == 2

    def test_capacity_one(self):
        """容量为 1 的极端情况"""
        stm = ShortTermMemory(capacity=1)
        stm.add(make_fragment(content="first", intensity=0.9))
        stm.add(make_fragment(content="second", intensity=0.9))
        assert len(stm) <= 1

    def test_evict_returns_fragment_or_none(self):
        """_evict 应返回被淘汰的 fragment（高重要度）或 None"""
        stm = ShortTermMemory(capacity=2)
        stm.add(make_fragment(content="重要", intensity=0.9, emotion=EmotionTag.LOVE))
        stm.add(make_fragment(content="普通", intensity=0.5))
        stm.add(make_fragment(content="触发淘汰", intensity=0.5))
        # _evict 被 add 内部调用；无法直接捕获返回值
        # 但可以验证容量约束
        assert len(stm) <= 2

    def test_exact_capacity(self):
        """恰好装满容量不应触发淘汰"""
        stm = ShortTermMemory(capacity=3)
        stm.add(make_fragment(content="a"))
        stm.add(make_fragment(content="b"))
        stm.add(make_fragment(content="c"))
        assert len(stm) == 3

    def test_get_recent_ordering(self):
        """get_recent 应按时间降序（最新在前）"""
        stm = ShortTermMemory(capacity=10)
        now = time.time()
        stm.add(make_fragment(content="old", timestamp=now - 100))
        stm.add(make_fragment(content="mid", timestamp=now - 50))
        stm.add(make_fragment(content="new", timestamp=now))
        recent = stm.get_recent(3)
        assert recent[0].content == "new"
        assert recent[1].content == "mid"
        assert recent[2].content == "old"


# ═════════════════════════════════════════════
# 4. LongTermMemory 边界测试
# ═════════════════════════════════════════════

class TestLongTermMemoryAdvanced:

    def test_empty_query(self):
        ltm = LongTermMemory()
        assert ltm.query() == []

    def test_query_by_source(self):
        ltm = LongTermMemory(capacity=100)
        ltm.add(make_fragment(content="照片", source=SourceType.PHOTO, intensity=0.8))
        ltm.add(make_fragment(content="对话", source=SourceType.CONVERSATION, intensity=0.8))
        results = ltm.query(source=SourceType.PHOTO)
        assert len(results) == 1
        assert results[0].source == SourceType.PHOTO

    def test_query_limit(self):
        ltm = LongTermMemory(capacity=100)
        for i in range(20):
            ltm.add(make_fragment(content=f"m{i}", intensity=0.8))
        results = ltm.query(limit=5)
        assert len(results) == 5

    def test_query_default_limit(self):
        """默认 limit=10"""
        ltm = LongTermMemory(capacity=100)
        for i in range(15):
            ltm.add(make_fragment(content=f"m{i}", intensity=0.8))
        results = ltm.query()
        assert len(results) == 10

    def test_query_combined_filters(self):
        ltm = LongTermMemory(capacity=100)
        ltm.add(make_fragment(content="A", source=SourceType.PHOTO, emotion=EmotionTag.JOY, intensity=0.9))
        ltm.add(make_fragment(content="B", source=SourceType.PHOTO, emotion=EmotionTag.SADNESS, intensity=0.9))
        ltm.add(make_fragment(content="C", source=SourceType.TIME, emotion=EmotionTag.JOY, intensity=0.9))
        results = ltm.query(emotion=EmotionTag.JOY, source=SourceType.PHOTO)
        assert len(results) == 1
        assert results[0].content == "A"

    def test_query_sorted_by_importance(self):
        ltm = LongTermMemory(capacity=100)
        ltm.add(make_fragment(content="弱", intensity=0.2))
        ltm.add(make_fragment(content="强", intensity=0.95))
        ltm.add(make_fragment(content="中", intensity=0.5))
        results = ltm.query(limit=3)
        for i in range(len(results) - 1):
            assert results[i].importance >= results[i + 1].importance

    def test_get_strongest_on_empty(self):
        ltm = LongTermMemory()
        assert ltm.get_strongest(5) == []

    def test_get_oldest(self):
        now = time.time()
        ltm = LongTermMemory(capacity=100)
        ltm.add(make_fragment(content="新", timestamp=now))
        ltm.add(make_fragment(content="旧", timestamp=now - 10000))
        ltm.add(make_fragment(content="中间", timestamp=now - 5000))
        oldest = ltm.get_oldest(2)
        assert oldest[0].content == "旧"
        assert oldest[1].content == "中间"

    def test_get_oldest_on_empty(self):
        ltm = LongTermMemory()
        assert ltm.get_oldest(5) == []

    def test_forget_keeps_stronger(self):
        """容量满时，_forget 应淘汰存活力最低的"""
        ltm = LongTermMemory(capacity=3)
        ltm.add(make_fragment(content="弱", intensity=0.1, emotion=EmotionTag.CALM))
        ltm.add(make_fragment(content="强", intensity=0.95, emotion=EmotionTag.LOVE))
        ltm.add(make_fragment(content="中", intensity=0.5, emotion=EmotionTag.JOY))
        # 触发遗忘
        ltm.add(make_fragment(content="新", intensity=0.8, emotion=EmotionTag.NOSTALGIA))
        assert len(ltm) <= 3
        # "弱" 最可能被淘汰
        contents = [m.content for m in ltm.memories]
        assert "强" in contents

    def test_emotion_bonus_forget_protection(self):
        """LOVE/SADNESS 的情感加成应让它们更难被淘汰"""
        ltm = LongTermMemory(capacity=2)
        ltm.add(make_fragment(content="冷静弱", intensity=0.3, emotion=EmotionTag.CALM))
        ltm.add(make_fragment(content="爱强", intensity=0.3, emotion=EmotionTag.LOVE))
        # 触发遗忘
        ltm.add(make_fragment(content="新", intensity=0.8))
        contents = [m.content for m in ltm.memories]
        # LOVE 有 1.3 加成，CALM 没有，所以 "冷静弱" 更可能被淘汰
        assert "爱强" in contents


# ═════════════════════════════════════════════
# 5. DistillationEngine 深层测试
# ═════════════════════════════════════════════

class TestDistillationEngineAdvanced:

    def test_emotion_boost_values(self):
        """所有情感都应在 EMOTION_BOOST 中"""
        for tag in EmotionTag:
            assert tag in DistillationEngine.EMOTION_BOOST, f"{tag.name} 缺失 EMOTION_BOOST"

    def test_evaluate_love_higher_than_calm(self):
        """LOVE (1.4) 应比 CALM (0.8) 评估更高"""
        f_love = make_fragment(emotion=EmotionTag.LOVE, intensity=0.5)
        f_calm = make_fragment(emotion=EmotionTag.CALM, intensity=0.5)
        assert DistillationEngine.evaluate(f_love) > DistillationEngine.evaluate(f_calm)

    def test_evaluate_anxiety_lower_than_base(self):
        """ANXIETY (0.9) 乘数 < 1.0，应降低评估值"""
        frag = make_fragment(emotion=EmotionTag.ANXIETY, intensity=0.5, decay_rate=0.0)
        val = DistillationEngine.evaluate(frag)
        assert val < 0.5  # base_value * 0.9

    def test_distill_empty_short_term(self):
        """空短期记忆不应报错"""
        stm = ShortTermMemory()
        ltm = LongTermMemory()
        count = DistillationEngine.distill(stm, ltm)
        assert count == 0
        assert len(stm) == 0
        assert len(ltm) == 0

    def test_distill_all_low_value(self):
        """全部低价值记忆不进入长期记忆"""
        stm = ShortTermMemory(capacity=20)
        ltm = LongTermMemory(capacity=200)
        for i in range(5):
            stm.add(make_fragment(content=f"低{i}", intensity=0.05, emotion=EmotionTag.CALM))
        count = DistillationEngine.distill(stm, ltm)
        assert count == 0
        assert len(ltm) == 0
        assert len(stm) == 5

    def test_distill_halves_decay_rate(self):
        """蒸馏后记忆的衰减率应减半"""
        stm = ShortTermMemory(capacity=20)
        ltm = LongTermMemory(capacity=200)
        frag = make_fragment(content="高价值", intensity=0.9, emotion=EmotionTag.LOVE, decay_rate=0.2)
        stm.add(frag)
        DistillationEngine.distill(stm, ltm)
        assert len(ltm) == 1
        assert ltm.memories[0].decay_rate == pytest.approx(0.1)

    def test_distill_touches_fragment(self):
        """蒸馏后 fragment 的 access_count 应增加"""
        stm = ShortTermMemory(capacity=20)
        ltm = LongTermMemory(capacity=200)
        frag = make_fragment(content="高价值", intensity=0.9, emotion=EmotionTag.LOVE)
        assert frag.access_count == 0
        stm.add(frag)
        DistillationEngine.distill(stm, ltm)
        assert ltm.memories[0].access_count >= 1

    def test_should_distill_high_value_count(self):
        """当有 >= 3 条高价值记忆时应触发蒸馏"""
        stm = ShortTermMemory(capacity=100)
        for i in range(3):
            stm.add(make_fragment(content=f"高{i}", intensity=0.9, emotion=EmotionTag.LOVE))
        # 虽然不满 80%，但高价值记忆 >= 3
        assert DistillationEngine.should_distill(stm) is True

    def test_should_not_distill_few_low_value(self):
        """少量低价值记忆不应触发蒸馏"""
        stm = ShortTermMemory(capacity=100)
        stm.add(make_fragment(intensity=0.1, emotion=EmotionTag.CALM))
        stm.add(make_fragment(intensity=0.1, emotion=EmotionTag.CALM))
        assert DistillationEngine.should_distill(stm) is False

    def test_should_distill_exactly_80_percent(self):
        """恰好 80% 容量应触发蒸馏"""
        stm = ShortTermMemory(capacity=10)
        for i in range(8):  # 8/10 = 80%
            stm.add(make_fragment(intensity=0.3))
        assert DistillationEngine.should_distill(stm) is True

    def test_should_distill_below_80_percent(self):
        """低于 80% 且高价值 < 3 不应触发"""
        stm = ShortTermMemory(capacity=10)
        for i in range(7):  # 7/10 = 70%
            stm.add(make_fragment(intensity=0.3))
        assert DistillationEngine.should_distill(stm) is False

    def test_distill_preserves_short_term_low_value(self):
        """低价值记忆应留在短期记忆中"""
        stm = ShortTermMemory(capacity=20)
        ltm = LongTermMemory(capacity=200)
        stm.add(make_fragment(content="低", intensity=0.05, emotion=EmotionTag.CALM))
        stm.add(make_fragment(content="高", intensity=0.9, emotion=EmotionTag.LOVE))
        DistillationEngine.distill(stm, ltm)
        remaining = [m.content for m in stm.memories]
        assert "低" in remaining


# ═════════════════════════════════════════════
# 6. ForgettingEngine 测试（完全未覆盖）
# ═════════════════════════════════════════════

class TestForgettingEngine:
    """ForgettingEngine 完整测试"""

    def test_forget_removes_weak_memories(self):
        """重要度低于阈值的记忆应被遗忘"""
        ltm = LongTermMemory(capacity=200)
        # 创建一条重要度极低的记忆（低强度 + 高衰减 + 老旧）
        now = time.time()
        weak = make_fragment(
            content="将被遗忘",
            intensity=0.1,
            emotion=EmotionTag.CALM,
            decay_rate=10.0,
            timestamp=now - 86400 * 365,  # 1年前
        )
        ltm.add(weak)
        count = ForgettingEngine.forget(ltm)
        assert count == 1
        assert len(ltm) == 0

    def test_forget_protects_love(self):
        """LOVE 情感的记忆受保护，即使重要度低"""
        ltm = LongTermMemory(capacity=200)
        now = time.time()
        protected = make_fragment(
            content="爱的记忆",
            intensity=0.01,
            emotion=EmotionTag.LOVE,
            decay_rate=10.0,
            timestamp=now - 86400 * 365,
        )
        ltm.add(protected)
        count = ForgettingEngine.forget(ltm)
        assert count == 0
        assert len(ltm) == 1

    def test_forget_protects_sadness(self):
        """SADNESS 情感的记忆受保护"""
        ltm = LongTermMemory(capacity=200)
        now = time.time()
        protected = make_fragment(
            content="悲伤的记忆",
            intensity=0.01,
            emotion=EmotionTag.SADNESS,
            decay_rate=10.0,
            timestamp=now - 86400 * 365,
        )
        ltm.add(protected)
        count = ForgettingEngine.forget(ltm)
        assert count == 0
        assert len(ltm) == 1

    def test_forget_protects_nostalgia(self):
        """NOSTALGIA 情感的记忆受保护"""
        ltm = LongTermMemory(capacity=200)
        now = time.time()
        protected = make_fragment(
            content="怀旧的记忆",
            intensity=0.01,
            emotion=EmotionTag.NOSTALGIA,
            decay_rate=10.0,
            timestamp=now - 86400 * 365,
        )
        ltm.add(protected)
        count = ForgettingEngine.forget(ltm)
        assert count == 0
        assert len(ltm) == 1

    def test_forget_on_empty(self):
        """空长期记忆不应报错"""
        ltm = LongTermMemory()
        count = ForgettingEngine.forget(ltm)
        assert count == 0

    def test_forget_keeps_strong_memories(self):
        """重要度高的记忆不应被遗忘"""
        ltm = LongTermMemory(capacity=200)
        ltm.add(make_fragment(content="强", intensity=0.9, emotion=EmotionTag.JOY))
        count = ForgettingEngine.forget(ltm)
        assert count == 0
        assert len(ltm) == 1

    def test_forget_mixed_memories(self):
        """混合强弱记忆，只遗忘弱的"""
        ltm = LongTermMemory(capacity=200)
        now = time.time()
        # 强记忆
        ltm.add(make_fragment(content="强", intensity=0.9, emotion=EmotionTag.JOY))
        # 弱记忆（非保护情感）
        weak = make_fragment(
            content="弱",
            intensity=0.01,
            emotion=EmotionTag.CALM,
            decay_rate=10.0,
            timestamp=now - 86400 * 365,
        )
        ltm.add(weak)
        count = ForgettingEngine.forget(ltm)
        assert count == 1
        assert len(ltm) == 1
        assert ltm.memories[0].content == "强"

    def test_forget_threshold_boundary(self):
        """重要度恰好等于阈值（0.05）不应被遗忘（< 才遗忘，== 不遗忘）"""
        ltm = LongTermMemory(capacity=200)
        frag = make_fragment(intensity=0.05, decay_rate=0.0, emotion=EmotionTag.CALM)
        ltm.add(frag)
        count = ForgettingEngine.forget(ltm)
        # importance == 0.05, threshold == 0.05, 条件是 importance < threshold
        # 0.05 < 0.05 为 False，所以不遗忘
        assert count == 0

    def test_get_fading_memories(self):
        """获取正在淡化的记忆"""
        ltm = LongTermMemory(capacity=200)
        now = time.time()
        # 淡化中的记忆（0.05 < importance < 0.2）
        fading = make_fragment(
            content="淡化中",
            intensity=0.15,
            emotion=EmotionTag.CALM,
            decay_rate=0.0,
        )
        # 健康的记忆
        healthy = make_fragment(
            content="健康",
            intensity=0.9,
            emotion=EmotionTag.JOY,
            decay_rate=0.0,
        )
        ltm.add(fading)
        ltm.add(healthy)
        result = ForgettingEngine.get_fading_memories(ltm, threshold=0.2)
        assert len(result) == 1
        assert result[0].content == "淡化中"

    def test_get_fading_memories_empty(self):
        ltm = LongTermMemory()
        assert ForgettingEngine.get_fading_memories(ltm) == []

    def test_get_fading_memories_custom_threshold(self):
        """自定义阈值"""
        ltm = LongTermMemory(capacity=200)
        frag = make_fragment(intensity=0.3, decay_rate=0.0)
        ltm.add(frag)
        # 默认 threshold=0.2，importance=0.3 > 0.2 → 不在淡化范围
        assert len(ForgettingEngine.get_fading_memories(ltm, threshold=0.2)) == 0
        # 提高 threshold
        assert len(ForgettingEngine.get_fading_memories(ltm, threshold=0.5)) == 1

    def test_protected_emotions_set(self):
        """验证受保护的情感集合"""
        assert ForgettingEngine.PROTECTED_EMOTIONS == {
            EmotionTag.LOVE,
            EmotionTag.SADNESS,
            EmotionTag.NOSTALGIA,
        }


# ═════════════════════════════════════════════
# 7. MemorySystem 集成测试（完全未覆盖）
# ═════════════════════════════════════════════

class TestMemorySystem:
    """MemorySystem 端到端集成测试"""

    def test_initial_state(self):
        system = MemorySystem()
        assert len(system.short_term) == 0
        assert len(system.long_term) == 0
        system.distillation_count == 0
        system.forget_count == 0

    def test_perceive_adds_to_short_term(self):
        system = MemorySystem()
        system.perceive("测试", SourceType.PHOTO, EmotionTag.JOY, 0.5)
        assert len(system.short_term) == 1

    def test_perceive_auto_distill(self):
        """当短期记忆接近满时，perceive 应自动触发蒸馏"""
        system = MemorySystem()
        # 填满短期记忆（capacity=20, 80% = 16）
        for i in range(18):
            system.perceive(f"记忆{i}", SourceType.TIME, EmotionTag.LOVE, 0.9)
        # 应该有蒸馏发生
        assert system.distillation_count >= 1
        assert len(system.long_term) >= 1

    def test_distill_moves_to_long_term(self):
        system = MemorySystem()
        system.perceive("重要", SourceType.PHOTO, EmotionTag.LOVE, 0.9)
        count = system.distill()
        assert count >= 1
        assert len(system.long_term) >= 1

    def test_forget_removes_from_long_term(self):
        system = MemorySystem()
        # 手动添加弱记忆到长期记忆
        now = time.time()
        weak = make_fragment(
            content="弱",
            intensity=0.01,
            emotion=EmotionTag.CALM,
            decay_rate=10.0,
            timestamp=now - 86400 * 365,
        )
        system.long_term.add(weak)
        count = system.forget()
        assert count == 1
        assert system.forget_count == 1

    def test_recall_returns_results(self):
        system = MemorySystem()
        system.perceive("回忆", SourceType.PHOTO, EmotionTag.NOSTALGIA, 0.9)
        system.distill()
        results = system.recall()
        assert len(results) >= 1

    def test_recall_by_emotion(self):
        system = MemorySystem()
        system.perceive("怀旧", SourceType.PHOTO, EmotionTag.NOSTALGIA, 0.9)
        system.perceive("开心", SourceType.PHOTO, EmotionTag.JOY, 0.9)
        system.distill()
        results = system.recall(emotion=EmotionTag.NOSTALGIA)
        assert all(m.emotion == EmotionTag.NOSTALGIA for m in results)

    def test_recall_touches_memories(self):
        """回忆应刷新记忆的访问计数"""
        system = MemorySystem()
        system.perceive("回忆", SourceType.PHOTO, EmotionTag.LOVE, 0.9)
        system.distill()
        mem = system.long_term.memories[0]
        old_count = mem.access_count
        system.recall()
        assert mem.access_count > old_count

    def test_status_returns_expected_keys(self):
        system = MemorySystem()
        status = system.status()
        expected_keys = {"短期记忆", "长期记忆", "蒸馏次数", "遗忘次数", "最强记忆", "正在淡化"}
        assert set(status.keys()) == expected_keys

    def test_status_values_type(self):
        system = MemorySystem()
        status = system.status()
        assert isinstance(status["短期记忆"], int)
        assert isinstance(status["长期记忆"], int)
        assert isinstance(status["蒸馏次数"], int)
        assert isinstance(status["遗忘次数"], int)
        assert isinstance(status["最强记忆"], list)
        assert isinstance(status["正在淡化"], int)

    def test_full_pipeline(self):
        """完整流水线：感知 → 蒸馏 → 回忆 → 遗忘"""
        system = MemorySystem()

        # 1. 感知一批记忆
        for i in range(10):
            system.perceive(
                f"感知{i}",
                SourceType.CONVERSATION,
                EmotionTag.LOVE,
                intensity=0.8,
            )

        # 2. 手动蒸馏
        distill_count = system.distill()
        assert distill_count >= 1

        # 3. 回忆
        results = system.recall()
        assert len(results) >= 1

        # 4. 遗忘
        forget_count = system.forget()
        # LOVE 受保护，不应被遗忘
        assert forget_count == 0

        # 5. 检查状态（perceive 会触发自动蒸馏，所以次数 >= 1）
        status = system.status()
        assert status["蒸馏次数"] >= 1
        assert status["遗忘次数"] >= 1

    def test_perceive_zero_intensity(self):
        """零强度感知不应崩溃"""
        system = MemorySystem()
        system.perceive("零强度", SourceType.TIME, EmotionTag.CALM, 0.0)
        assert len(system.short_term) == 1

    def test_perceive_max_intensity(self):
        """最大强度感知"""
        system = MemorySystem()
        system.perceive("最大", SourceType.PHOTO, EmotionTag.LOVE, 1.0)
        assert len(system.short_term) == 1

    def test_multiple_distill_cycles(self):
        """多次蒸馏循环"""
        system = MemorySystem()

        # 第一轮感知 + 蒸馏
        for i in range(10):
            system.perceive(f"轮1-{i}", SourceType.PHOTO, EmotionTag.JOY, 0.8)
        system.distill()

        # 第二轮感知 + 蒸馏
        for i in range(10):
            system.perceive(f"轮2-{i}", SourceType.TIME, EmotionTag.SADNESS, 0.7)
        system.distill()

        # perceive 会触发自动蒸馏，加上手动调用，总次数 >= 2
        assert system.distillation_count >= 2
        assert len(system.long_term) >= 2

    def test_recall_on_empty_system(self):
        """空系统的回忆不应崩溃"""
        system = MemorySystem()
        results = system.recall()
        assert results == []

    def test_recall_query_matching(self):
        """recall 的 query 参数支持关键词匹配"""
        system = MemorySystem()
        system.perceive("夕阳很美", SourceType.PHOTO, EmotionTag.JOY, 0.9)
        system.perceive("下雨了", SourceType.WEATHER, EmotionTag.CALM, 0.5)
        system.distill()
        # 匹配关键词
        results = system.recall(query="夕阳")
        assert len(results) >= 1
        assert any("夕阳" in r.content for r in results)
        # 不匹配的关键词返回空
        results = system.recall(query="不存在的关键词")
        assert len(results) == 0

    def test_forget_count_increments(self):
        system = MemorySystem()
        system.forget()
        system.forget()
        assert system.forget_count == 2

    def test_distill_count_increments(self):
        system = MemorySystem()
        system.perceive("a", SourceType.PHOTO, EmotionTag.LOVE, 0.9)
        system.distill()
        system.perceive("b", SourceType.PHOTO, EmotionTag.LOVE, 0.9)
        system.distill()
        assert system.distillation_count == 2


# ═════════════════════════════════════════════
# 8. EmotionTag / SourceType 枚举完整性
# ═════════════════════════════════════════════

class TestEnums:

    def test_emotion_tag_count(self):
        assert len(EmotionTag) == 10

    def test_source_type_count(self):
        assert len(SourceType) == 7

    def test_emotion_tag_values_are_chinese(self):
        """所有情感标签的值应为中文"""
        for tag in EmotionTag:
            assert any('一' <= c <= '鿿' for c in tag.value), \
                f"{tag.name} = '{tag.value}' 不是中文"

    def test_source_type_values_are_chinese(self):
        for tag in SourceType:
            assert any('一' <= c <= '鿿' for c in tag.value), \
                f"{tag.name} = '{tag.value}' 不是中文"

    def test_emotion_tag_unique_values(self):
        values = [tag.value for tag in EmotionTag]
        assert len(values) == len(set(values))

    def test_source_type_unique_values(self):
        values = [tag.value for tag in SourceType]
        assert len(values) == len(set(values))

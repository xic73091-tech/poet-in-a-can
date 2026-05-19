"""记忆蒸馏算法的单元测试"""

import sys
import os
import importlib
import pytest

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 将 02_生命机制 目录加入 sys.path，以便按模块名导入
sys.path.insert(0, os.path.join(PROJECT_ROOT, "02_生命机制"))

# 使用 importlib 导入含中文名的模块
memory_module = importlib.import_module("记忆蒸馏算法")

MemoryFragment = memory_module.MemoryFragment
ShortTermMemory = memory_module.ShortTermMemory
LongTermMemory = memory_module.LongTermMemory
DistillationEngine = memory_module.DistillationEngine
EmotionTag = memory_module.EmotionTag
SourceType = memory_module.SourceType


class TestMemoryFragment:
    """MemoryFragment 数据类测试"""

    def test_create_fragment(self):
        fragment = MemoryFragment(
            content="用户拍了一张夕阳的照片",
            source=SourceType.PHOTO,
            emotion=EmotionTag.NOSTALGIA,
            intensity=0.8,
        )
        assert fragment.content == "用户拍了一张夕阳的照片"
        assert fragment.source == SourceType.PHOTO
        assert fragment.emotion == EmotionTag.NOSTALGIA
        assert fragment.intensity == 0.8
        assert fragment.access_count == 0

    def test_touch_increments_access(self):
        fragment = MemoryFragment(
            content="测试",
            source=SourceType.CONVERSATION,
            emotion=EmotionTag.JOY,
            intensity=0.5,
        )
        assert fragment.access_count == 0
        fragment.touch()
        assert fragment.access_count == 1
        fragment.touch()
        assert fragment.access_count == 2

    def test_importance_positive(self):
        fragment = MemoryFragment(
            content="测试",
            source=SourceType.TIME,
            emotion=EmotionTag.CALM,
            intensity=0.6,
        )
        assert fragment.importance > 0

    def test_importance_increases_with_access(self):
        f1 = MemoryFragment(
            content="未访问",
            source=SourceType.TIME,
            emotion=EmotionTag.CALM,
            intensity=0.5,
        )
        f2 = MemoryFragment(
            content="已访问",
            source=SourceType.TIME,
            emotion=EmotionTag.CALM,
            intensity=0.5,
        )
        for _ in range(5):
            f2.touch()
        assert f2.importance > f1.importance


class TestShortTermMemory:
    """短期记忆测试"""

    def test_add_and_len(self):
        stm = ShortTermMemory(capacity=5)
        assert len(stm) == 0
        stm.add(MemoryFragment("a", SourceType.PHOTO, EmotionTag.JOY, 0.5))
        assert len(stm) == 1

    def test_capacity_eviction(self):
        stm = ShortTermMemory(capacity=3)
        for i in range(5):
            stm.add(MemoryFragment(f"记忆{i}", SourceType.TIME, EmotionTag.CALM, 0.5))
        assert len(stm) <= 3

    def test_get_recent(self):
        stm = ShortTermMemory(capacity=10)
        stm.add(MemoryFragment("旧", SourceType.TIME, EmotionTag.CALM, 0.3))
        stm.add(MemoryFragment("新", SourceType.TIME, EmotionTag.JOY, 0.9))
        recent = stm.get_recent(1)
        assert len(recent) == 1
        assert recent[0].content == "新"

    def test_get_by_emotion(self):
        stm = ShortTermMemory(capacity=10)
        stm.add(MemoryFragment("开心", SourceType.TIME, EmotionTag.JOY, 0.8))
        stm.add(MemoryFragment("平静", SourceType.TIME, EmotionTag.CALM, 0.4))
        stm.add(MemoryFragment("也开心", SourceType.TIME, EmotionTag.JOY, 0.7))
        joy_memories = stm.get_by_emotion(EmotionTag.JOY)
        assert len(joy_memories) == 2


class TestLongTermMemory:
    """长期记忆测试"""

    def test_add_and_query(self):
        ltm = LongTermMemory(capacity=100)
        ltm.add(MemoryFragment("回忆1", SourceType.PHOTO, EmotionTag.NOSTALGIA, 0.8))
        ltm.add(MemoryFragment("回忆2", SourceType.CONVERSATION, EmotionTag.LOVE, 0.9))
        results = ltm.query(emotion=EmotionTag.NOSTALGIA)
        assert len(results) == 1
        assert results[0].content == "回忆1"

    def test_query_with_min_importance(self):
        ltm = LongTermMemory(capacity=100)
        ltm.add(MemoryFragment("弱", SourceType.TIME, EmotionTag.CALM, 0.1))
        ltm.add(MemoryFragment("强", SourceType.TIME, EmotionTag.JOY, 0.9))
        results = ltm.query(min_importance=0.5)
        assert all(r.importance >= 0.5 for r in results)

    def test_get_strongest(self):
        ltm = LongTermMemory(capacity=100)
        ltm.add(MemoryFragment("弱记忆", SourceType.TIME, EmotionTag.CALM, 0.1))
        ltm.add(MemoryFragment("强记忆", SourceType.TIME, EmotionTag.LOVE, 0.95))
        strongest = ltm.get_strongest(1)
        assert len(strongest) == 1
        assert strongest[0].content == "强记忆"


class TestDistillationEngine:
    """蒸馏引擎测试"""

    def test_evaluate_returns_positive(self):
        fragment = MemoryFragment("测试", SourceType.PHOTO, EmotionTag.JOY, 0.7)
        value = DistillationEngine.evaluate(fragment)
        assert value > 0

    def test_distill_moves_high_value_memories(self):
        stm = ShortTermMemory(capacity=20)
        ltm = LongTermMemory(capacity=200)

        # 添加高价值记忆
        stm.add(MemoryFragment("重要", SourceType.PHOTO, EmotionTag.LOVE, 0.9))
        # 添加低价值记忆
        stm.add(MemoryFragment("琐碎", SourceType.TIME, EmotionTag.CALM, 0.1))

        stm_count_before = len(stm)
        distill_count = DistillationEngine.distill(stm, ltm)

        # 高价值记忆应被蒸馏到长期记忆
        assert distill_count >= 1
        assert len(ltm) >= 1
        assert len(stm) < stm_count_before

    def test_should_distill_near_capacity(self):
        stm = ShortTermMemory(capacity=10)
        # 填满到 80% 以上
        for i in range(9):
            stm.add(MemoryFragment(f"m{i}", SourceType.TIME, EmotionTag.CALM, 0.5))
        assert DistillationEngine.should_distill(stm) is True

    def test_should_not_distill_when_empty(self):
        stm = ShortTermMemory(capacity=20)
        assert DistillationEngine.should_distill(stm) is False

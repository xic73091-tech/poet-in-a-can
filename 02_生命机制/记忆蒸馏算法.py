"""
记忆蒸馏算法 - 罐头里的诗人

短期记忆 → 蒸馏 → 长期记忆 → 遗忘

核心逻辑：
- 短期记忆是感知和对话的原始记录，有容量上限
- 蒸馏是从短期记忆中提取有意义片段的过程
- 长期记忆是经过筛选的重要记忆，有情感标签
- 遗忘是基于时间和访问频率的记忆淡化
"""

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EmotionTag(Enum):
    """情感标签（对应 API 层英文标识：joy, sadness, calm, excitement, nostalgia, anxiety, love, loneliness, curiosity, melancholy）"""
    JOY = "喜悦"
    SADNESS = "悲伤"
    CALM = "平静"
    EXCITEMENT = "兴奋"
    NOSTALGIA = "怀旧"
    ANXIETY = "焦虑"
    LOVE = "爱"
    LONELINESS = "孤独"
    CURIOSITY = "好奇"
    MELANCHOLY = "忧郁"


class SourceType(Enum):
    """记忆来源（对应 API 层英文标识：photo, location, conversation, weather, time, gesture, silence）"""
    PHOTO = "照片"
    LOCATION = "位置"
    CONVERSATION = "对话"
    WEATHER = "天气"
    TIME = "时间"
    GESTURE = "手势"
    SILENCE = "沉默"


# 枚举值映射：内部中文 ↔ API英文（参考 05_技术架构/API接口定义.yaml）
EMOTION_API_MAP = {
    "喜悦": "joy", "悲伤": "sadness", "平静": "calm", "兴奋": "excitement",
    "怀旧": "nostalgia", "焦虑": "anxiety", "爱": "love", "孤独": "loneliness",
    "好奇": "curiosity", "忧郁": "melancholy",
}
SOURCE_API_MAP = {
    "照片": "photo", "位置": "location", "对话": "conversation",
    "天气": "weather", "时间": "time", "手势": "gesture", "沉默": "silence",
}
API_EMOTION_MAP = {v: k for k, v in EMOTION_API_MAP.items()}
API_SOURCE_MAP = {v: k for k, v in SOURCE_API_MAP.items()}


@dataclass
class MemoryFragment:
    """记忆片段"""
    content: str                          # 内容描述
    source: SourceType                    # 来源类型
    emotion: EmotionTag                   # 情感标签
    intensity: float                      # 情感强度 0-1
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0                 # 被访问次数
    last_accessed: float = 0              # 最后访问时间
    decay_rate: float = 0.1               # 衰减速率

    @property
    def age(self) -> float:
        """记忆年龄（秒）"""
        return time.time() - self.timestamp

    @property
    def importance(self) -> float:
        """重要度 = 情感强度 × (1 + 访问加成) × 时间衰减"""
        time_decay = math.exp(-self.decay_rate * self.age / 86400)  # 按天衰减
        access_bonus = min(self.access_count * 0.1, 0.5)  # 访问加成，上限0.5
        return self.intensity * (1 + access_bonus) * time_decay

    def touch(self):
        """访问记忆（刷新访问计数）"""
        self.access_count += 1
        self.last_accessed = time.time()


class ShortTermMemory:
    """
    短期记忆

    特点：
    - 容量有限（默认20条）
    - 存储原始感知和对话
    - 满了之后最旧的被淘汰或蒸馏
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.memories: list[MemoryFragment] = []

    def add(self, fragment: MemoryFragment) -> Optional[MemoryFragment]:
        """添加记忆片段，返回被驱逐的高价值记忆（供蒸馏引擎处理）"""
        self.memories.append(fragment)
        evicted = None
        if len(self.memories) > self.capacity:
            evicted = self._evict()
        return evicted

    def _evict(self):
        """淘汰最不重要的记忆"""
        if not self.memories:
            return
        # 按重要度排序，淘汰最低的
        self.memories.sort(key=lambda m: m.importance, reverse=True)
        evicted = self.memories.pop()
        # 如果重要度足够高，进入蒸馏候选
        if evicted.importance > 0.3:
            return evicted  # 返回给蒸馏引擎处理
        return None  # 直接遗忘

    def get_recent(self, n: int = 5) -> list[MemoryFragment]:
        """获取最近n条记忆"""
        return sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:n]

    def get_by_emotion(self, emotion: EmotionTag) -> list[MemoryFragment]:
        """按情感标签筛选"""
        return [m for m in self.memories if m.emotion == emotion]

    def __len__(self):
        return len(self.memories)


class LongTermMemory:
    """
    长期记忆

    特点：
    - 容量较大（默认200条）
    - 存储蒸馏后的重要记忆
    - 有遗忘机制
    - 支持按情感、来源、时间查询
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self.memories: list[MemoryFragment] = []

    def add(self, fragment: MemoryFragment):
        """添加蒸馏后的记忆"""
        self.memories.append(fragment)
        if len(self.memories) > self.capacity:
            self._forget()

    def _forget(self):
        """遗忘最弱的记忆"""
        if not self.memories:
            return
        # 计算每条记忆的存活力
        survival_scores = []
        for m in self.memories:
            # 存活力 = 重要度 × (1 + 情感极性加成)
            emotion_bonus = 1.0
            if m.emotion in (EmotionTag.JOY, EmotionTag.LOVE, EmotionTag.SADNESS):
                emotion_bonus = 1.3  # 强烈情感更难遗忘
            survival = m.importance * emotion_bonus
            survival_scores.append((survival, m))

        survival_scores.sort(key=lambda x: x[0])
        # 遗忘存活力最低的
        to_forget = survival_scores[0][1]
        self.memories.remove(to_forget)

    def query(
        self,
        emotion: Optional[EmotionTag] = None,
        source: Optional[SourceType] = None,
        min_importance: float = 0,
        limit: int = 10
    ) -> list[MemoryFragment]:
        """查询长期记忆"""
        results = self.memories
        if emotion:
            results = [m for m in results if m.emotion == emotion]
        if source:
            results = [m for m in results if m.source == source]
        results = [m for m in results if m.importance >= min_importance]
        results.sort(key=lambda m: m.importance, reverse=True)
        return results[:limit]

    def get_strongest(self, n: int = 5) -> list[MemoryFragment]:
        """获取最强记忆（最重要的）"""
        return sorted(self.memories, key=lambda m: m.importance, reverse=True)[:n]

    def get_oldest(self, n: int = 5) -> list[MemoryFragment]:
        """获取最久远的记忆"""
        return sorted(self.memories, key=lambda m: m.timestamp)[:n]

    def __len__(self):
        return len(self.memories)


class DistillationEngine:
    """
    蒸馏引擎

    职责：
    - 从短期记忆中提取有意义的片段
    - 为片段添加情感标签和重要度
    - 决定哪些记忆值得进入长期存储
    """

    # 重要度阈值：超过此值的记忆才会被蒸馏
    IMPORTANCE_THRESHOLD = 0.4

    # 情感强度加成规则
    EMOTION_BOOST = {
        EmotionTag.JOY: 1.2,
        EmotionTag.SADNESS: 1.3,
        EmotionTag.LOVE: 1.4,
        EmotionTag.LONELINESS: 1.1,
        EmotionTag.NOSTALGIA: 1.2,
        EmotionTag.ANXIETY: 0.9,
        EmotionTag.CALM: 0.8,
        EmotionTag.EXCITEMENT: 1.1,
        EmotionTag.CURIOSITY: 1.0,
        EmotionTag.MELANCHOLY: 1.2,
    }

    @classmethod
    def evaluate(cls, fragment: MemoryFragment) -> float:
        """评估记忆片段的蒸馏价值"""
        base_value = fragment.importance
        emotion_multiplier = cls.EMOTION_BOOST.get(fragment.emotion, 1.0)
        return base_value * emotion_multiplier

    @classmethod
    def distill(cls, short_term: ShortTermMemory, long_term: LongTermMemory):
        """
        执行一次蒸馏

        流程：
        1. 评估短期记忆中每条记忆的价值
        2. 价值超过阈值的记忆进入长期存储
        3. 被蒸馏的记忆从短期记忆中移除
        """
        to_keep = []
        to_distill = []

        for memory in short_term.memories:
            value = cls.evaluate(memory)
            if value >= cls.IMPORTANCE_THRESHOLD:
                # 增加衰减速率（长期记忆衰减更慢）
                memory.decay_rate = memory.decay_rate * 0.5
                to_distill.append(memory)
            else:
                to_keep.append(memory)

        # 蒸馏到长期记忆
        for memory in to_distill:
            memory.touch()  # 标记为已访问
            long_term.add(memory)

        # 更新短期记忆
        short_term.memories = to_keep

        return len(to_distill)

    @classmethod
    def should_distill(cls, short_term: ShortTermMemory) -> bool:
        """判断是否需要执行蒸馏"""
        # 条件1：短期记忆接近容量上限
        if len(short_term) >= short_term.capacity * 0.8:
            return True
        # 条件2：有高价值记忆等待蒸馏
        high_value_count = sum(
            1 for m in short_term.memories
            if cls.evaluate(m) >= cls.IMPORTANCE_THRESHOLD
        )
        if high_value_count >= 3:
            return True
        return False


class ForgettingEngine:
    """
    遗忘引擎

    职责：
    - 定期清理长期记忆中已经淡化的记忆
    - 模拟人类的自然遗忘过程
    - 保护重要记忆不被遗忘
    """

    # 遗忘阈值：重要度低于此值的记忆将被遗忘
    FORGET_THRESHOLD = 0.05

    # 受保护的情感：这些情感的记忆不容易被遗忘
    PROTECTED_EMOTIONS = {
        EmotionTag.LOVE,
        EmotionTag.SADNESS,
        EmotionTag.NOSTALGIA,
    }

    @classmethod
    def forget(cls, long_term: LongTermMemory) -> int:
        """
        执行一次遗忘

        返回遗忘的记忆数量
        """
        to_forget = []
        to_keep = []

        for memory in long_term.memories:
            # 受保护的记忆不遗忘
            if memory.emotion in cls.PROTECTED_EMOTIONS:
                to_keep.append(memory)
                continue

            # 重要度低于阈值的记忆被遗忘
            if memory.importance < cls.FORGET_THRESHOLD:
                to_forget.append(memory)
            else:
                to_keep.append(memory)

        long_term.memories = to_keep
        return len(to_forget)

    @classmethod
    def get_fading_memories(cls, long_term: LongTermMemory, threshold: float = 0.2) -> list[MemoryFragment]:
        """获取正在淡化的记忆（重要度低于threshold但高于遗忘阈值）"""
        return [
            m for m in long_term.memories
            if cls.FORGET_THRESHOLD < m.importance < threshold
        ]


class MemorySystem:
    """
    记忆系统总控

    整合短期记忆、长期记忆、蒸馏引擎和遗忘引擎
    """

    def __init__(self):
        self.short_term = ShortTermMemory(capacity=20)
        self.long_term = LongTermMemory(capacity=200)
        self.distillation_count = 0
        self.forget_count = 0

    def perceive(self, content: str, source: SourceType, emotion: EmotionTag, intensity: float):
        """记录一次感知"""
        fragment = MemoryFragment(
            content=content,
            source=source,
            emotion=emotion,
            intensity=intensity
        )
        evicted = self.short_term.add(fragment)

        # 被驱逐的高价值记忆直接进入长期记忆（避免丢失）
        if evicted is not None:
            evicted.touch()
            self.long_term.add(evicted)

        # 检查是否需要蒸馏
        if DistillationEngine.should_distill(self.short_term):
            self.distill()

    def distill(self) -> int:
        """执行蒸馏"""
        count = DistillationEngine.distill(self.short_term, self.long_term)
        self.distillation_count += 1
        return count

    def forget(self) -> int:
        """执行遗忘"""
        count = ForgettingEngine.forget(self.long_term)
        self.forget_count += 1
        return count

    def recall(self, query: str = "", emotion: Optional[EmotionTag] = None) -> list[MemoryFragment]:
        """回忆：从长期记忆中检索，支持关键词和情感过滤"""
        if query:
            # 关键词匹配：在内容中搜索
            candidates = [m for m in self.long_term.memories if query in m.content]
            if emotion:
                candidates = [m for m in candidates if m.emotion == emotion]
            candidates.sort(key=lambda m: m.importance, reverse=True)
            results = candidates[:5]
        else:
            results = self.long_term.query(emotion=emotion, limit=5)
        # 访问这些记忆（延长它们的寿命）
        for m in results:
            m.touch()
        return results

    def status(self) -> dict:
        """系统状态"""
        return {
            "短期记忆": len(self.short_term),
            "长期记忆": len(self.long_term),
            "蒸馏次数": self.distillation_count,
            "遗忘次数": self.forget_count,
            "最强记忆": self.long_term.get_strongest(1),
            "正在淡化": len(ForgettingEngine.get_fading_memories(self.long_term)),
        }


# ===== 使用示例 =====

if __name__ == "__main__":
    system = MemorySystem()

    # 模拟一天的感知
    system.perceive(
        "用户拍了一张夕阳的照片，天空是橙色的",
        SourceType.PHOTO,
        EmotionTag.NOSTALGIA,
        intensity=0.8
    )

    system.perceive(
        "用户在深夜打开手机，屏幕亮度很低",
        SourceType.TIME,
        EmotionTag.LONELINESS,
        intensity=0.6
    )

    system.perceive(
        "用户分享了一首歌，是老歌",
        SourceType.CONVERSATION,
        EmotionTag.NOSTALGIA,
        intensity=0.7
    )

    system.perceive(
        "今天下雨了，用户没有出门",
        SourceType.WEATHER,
        EmotionTag.CALM,
        intensity=0.4
    )

    system.perceive(
        "用户给妈妈打了电话，聊了半小时",
        SourceType.CONVERSATION,
        EmotionTag.LOVE,
        intensity=0.9
    )

    # 查看状态
    print("=== 记忆系统状态 ===")
    status = system.status()
    for key, value in status.items():
        if key == "最强记忆":
            print(f"{key}: {value[0].content if value else '无'}")
        else:
            print(f"{key}: {value}")

    # 回忆怀旧的记忆
    print("\n=== 怀旧的记忆 ===")
    nostalgic = system.recall(emotion=EmotionTag.NOSTALGIA)
    for m in nostalgic:
        print(f"[{m.source.value}] {m.content} (重要度: {m.importance:.2f})")

    # 模拟时间流逝（压缩到几秒内测试）
    print("\n=== 蒸馏后 ===")
    system.distill()
    status = system.status()
    print(f"短期记忆: {status['短期记忆']}")
    print(f"长期记忆: {status['长期记忆']}")

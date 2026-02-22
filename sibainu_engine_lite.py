import torch
import torch.nn.functional as F

class SibainuEngineLite:
    def __init__(self, threshold=0.85):
        """
        Sibainu Engine Lite Edition
        解析軸: Anchor Drift (幾何学的乖離)
        """
        self.threshold = threshold

    def calculate_drift(self, current_hidden_state, anchor_hidden_state):
        """
        各トークンのHidden Stateとプロンプトアンカーとの幾何学的距離を測定
        計算量: O(d)
        """
        # コサイン類似度をベースにした乖離スコアの計算
        # 1.0 (完全一致) -> 0.0 (無相関)
        similarity = F.cosine_similarity(current_hidden_state, anchor_hidden_state, dim=-1)
        drift_score = 1.0 - similarity.item()
        return drift_score

    def detect(self, score):
        """
        閾値に基づいたハルシネーション検知
        """
        if score > self.threshold:
            return True, "ALERT: Geometric anomaly detected."
        return False, "PASS: Stable."
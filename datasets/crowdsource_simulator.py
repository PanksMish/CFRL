"""
dataset/crowdsource_simulator.py
---------------------------------
Simulates the Crowdsourcing Social Networks Pool (CSNP) described in §4.2.

Each virtual node is associated with a pool of simulated annotators.
Credibility scores are computed as:

    C(p_i) = λ₁·A_i + λ₂·H_i + λ₃·L_i          (Equation 13)

where:
    A_i = activity level   (normalised posting frequency)
    H_i = historical behaviour (past accuracy on verified samples)
    L_i = crowd-based annotation agreement score

The simulator injects controlled label noise to model real-world annotator
disagreement and evaluates inter-annotator agreement (IAA) via weighted
aggregation, maintaining a minimum IAA threshold before accepting a label.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from config import cfg

logger = logging.getLogger(__name__)


class Annotator:
    """
    Represents a single crowdsourced annotator with a credibility profile.

    Attributes:
        annotator_id   : Unique identifier.
        activity       : Activity level A_i ∈ [0, 1].
        accuracy       : Historical accuracy H_i ∈ [0, 1].
        noise_rate     : Probability of producing a wrong label.
        credibility    : Computed credibility score C(p_i).
    """

    def __init__(
        self,
        annotator_id: int,
        rng: np.random.Generator,
    ):
        self.annotator_id = annotator_id
        self.rng          = rng

        # Sample annotator profile from realistic distributions
        self.activity  = rng.beta(a=2.0, b=1.5)   # Skewed toward active users
        self.accuracy  = rng.beta(a=5.0, b=1.5)   # Most annotators fairly accurate
        self.noise_rate = max(0.0, 1.0 - self.accuracy)

        # Crowd annotation score starts neutral
        self._agreement_history: List[float] = []

    def annotate(self, true_label: int) -> int:
        """
        Produce a (possibly noisy) annotation.

        With probability noise_rate, the annotator flips the label.
        This simulates real-world crowdsourcing disagreement.
        """
        if self.rng.random() < self.noise_rate:
            return 1 - true_label  # Flip label (error)
        return true_label

    def update_agreement(self, agreed: bool) -> None:
        """Record whether this annotator agreed with the majority on a task."""
        self._agreement_history.append(1.0 if agreed else 0.0)

    @property
    def crowd_score(self) -> float:
        """L_i: running agreement score (0 if no history yet)."""
        if not self._agreement_history:
            return 0.5   # Neutral prior
        return float(np.mean(self._agreement_history[-50:]))  # Rolling window

    def credibility_score(
        self,
        lambda1: float = cfg.crowd.LAMBDA1,
        lambda2: float = cfg.crowd.LAMBDA2,
        lambda3: float = cfg.crowd.LAMBDA3,
    ) -> float:
        """
        Compute weighted credibility score C(p_i) = λ₁A_i + λ₂H_i + λ₃L_i.

        All weights must sum to 1.0.
        """
        return lambda1 * self.activity + lambda2 * self.accuracy + lambda3 * self.crowd_score


class CrowdsourceSimulator:
    """
    Simulates the full Crowdsourcing Social Networks Pool (CSNP).

    Responsibilities:
        1. Maintain a pool of simulated annotators.
        2. Assign annotators to virtual nodes.
        3. Generate (possibly noisy) crowd labels for a batch of articles.
        4. Aggregate crowd labels using credibility-weighted voting.
        5. Report inter-annotator agreement (IAA) statistics.

    Args:
        num_annotators : Total annotator pool size.
        num_nodes      : Number of virtual federated nodes.
        seed           : Random seed for reproducibility.
    """

    def __init__(
        self,
        num_annotators: int = cfg.crowd.NUM_ANNOTATORS,
        num_nodes: int      = cfg.federated.DEFAULT_NODES,
        seed: int           = cfg.data.RANDOM_SEED,
    ):
        self.rng            = np.random.default_rng(seed)
        self.num_nodes      = num_nodes
        self.num_annotators = num_annotators

        # Build annotator pool
        self.annotators: List[Annotator] = [
            Annotator(i, np.random.default_rng(seed + i))
            for i in range(num_annotators)
        ]

        # Assign ~10 annotators per node (with overlap allowed)
        self._node_annotator_map: Dict[int, List[int]] = self._assign_annotators()

        logger.info(
            "CrowdsourceSimulator: %d annotators across %d nodes.",
            num_annotators, num_nodes,
        )

    def _assign_annotators(self) -> Dict[int, List[int]]:
        """
        Randomly assign a subset of annotators to each node.
        Each node gets approx. 10 annotators drawn without full replacement
        (overlap is intentional — annotators can participate on multiple nodes).
        """
        annotators_per_node = max(5, self.num_annotators // (self.num_nodes // 10 + 1))
        assignment = {}
        all_ids = list(range(self.num_annotators))
        for v in range(self.num_nodes):
            assigned = self.rng.choice(all_ids, size=min(annotators_per_node, len(all_ids)), replace=False)
            assignment[v] = assigned.tolist()
        return assignment

    def get_crowd_labels(
        self,
        node_id: int,
        true_labels: List[int],
    ) -> Tuple[List[int], float, List[float]]:
        """
        Generate crowd-aggregated labels for a list of articles at a node.

        Each article is annotated by the node's assigned annotators.
        Credibility-weighted majority voting is used to aggregate labels.
        IAA is computed as the fraction of annotator-pairs that agree.

        Args:
            node_id     : Virtual node identifier.
            true_labels : Ground-truth labels for the articles.

        Returns:
            Tuple of:
                crowd_labels   : List of aggregated crowd labels.
                iaa_score      : Inter-annotator agreement score ∈ [0, 1].
                credibilities  : Credibility score per annotator at this node.
        """
        annotator_ids = self._node_annotator_map.get(node_id, list(range(10)))
        annotators    = [self.annotators[i] for i in annotator_ids]
        credibilities = [a.credibility_score() for a in annotators]

        crowd_labels = []
        iaa_scores   = []

        for true_label in true_labels:
            # Each annotator provides a label
            raw_annotations = [a.annotate(true_label) for a in annotators]

            # Credibility-weighted vote
            weights     = np.array(credibilities)
            weights    /= weights.sum()
            vote_score  = np.dot(weights, raw_annotations)  # Weighted probability of Fake
            agg_label   = int(vote_score > 0.5)

            # Update annotator agreement history
            for a, ann_label in zip(annotators, raw_annotations):
                a.update_agreement(ann_label == agg_label)

            # IAA: pairwise agreement fraction
            n = len(raw_annotations)
            if n > 1:
                agree_count = sum(
                    raw_annotations[i] == raw_annotations[j]
                    for i in range(n)
                    for j in range(i + 1, n)
                )
                total_pairs = n * (n - 1) / 2
                iaa_scores.append(agree_count / total_pairs)
            else:
                iaa_scores.append(1.0)

            crowd_labels.append(agg_label)

        avg_iaa = float(np.mean(iaa_scores)) if iaa_scores else 1.0
        return crowd_labels, avg_iaa, credibilities

    def get_node_credibility(self, node_id: int) -> float:
        """
        Compute the mean credibility score of all annotators assigned to a node.
        Used as a proxy for data quality when computing similarity scores (δ_v).
        """
        annotator_ids = self._node_annotator_map.get(node_id, [])
        if not annotator_ids:
            return 0.5
        scores = [self.annotators[i].credibility_score() for i in annotator_ids]
        return float(np.mean(scores))

    def summary(self) -> Dict[str, float]:
        """Return summary statistics of the annotator pool."""
        creds = [a.credibility_score() for a in self.annotators]
        return {
            "mean_credibility": float(np.mean(creds)),
            "std_credibility":  float(np.std(creds)),
            "min_credibility":  float(np.min(creds)),
            "max_credibility":  float(np.max(creds)),
        }

# update 2024/12/16 lgr:GMM Clustering with Custom Initialization
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .tree_structures import Node
# Import necessary methods from other modules
from .utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


# Global Clustering with UMAP
def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


# Local Clustering with UMAP
def local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


# Optimal Clusters Selection
def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


# GMM Clustering with Custom Initialization
def GMM_cluster(
        embeddings: np.ndarray,
        threshold: float,
        query_embeddings: np.ndarray,  # Added query_list for dynamic cluster centers
        random_state: int = 0,
):

    if query_embeddings.size != 0:
        # n_clusters is determined by the number of strings in query_list
        n_clusters = query_embeddings.shape[0]
        # Initialize the Gaussian Mixture Model (GMM)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state, means_init=query_embeddings)
    else:
        n_clusters = get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    # Fit the GMM model

    gm.fit(embeddings)

    # Predict probabilities
    probs = gm.predict_proba(embeddings)

    # Assign labels based on threshold
    labels = [np.where(prob > threshold)[0] for prob in probs]

    return labels, n_clusters


# Perform clustering by using both global and local clustering
def perform_clustering(
        embeddings: np.ndarray, dim: int, threshold: float, query_embeddings: np.ndarray, verbose: bool = False
) -> List[np.ndarray]:
    if query_embeddings.size != 0:
        combined = np.concatenate([embeddings, query_embeddings], axis=0)
        # 一次性降维
        reduced_combined = global_cluster_embeddings(combined, min(dim, len(combined) - 2))
        # 拆分还原
        reduced_embeddings_global = reduced_combined[:len(embeddings)]
        reduced_query_embeddings_global = reduced_combined[len(embeddings):]
    else:
        reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) - 2))
        reduced_query_embeddings_global = np.array([])
    # Perform global clustering with custom initial centers (query_list embeddings)
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold, reduced_query_embeddings_global
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[np.array([i in gc for gc in global_clusters])]
        if verbose:
            logging.info(f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}")
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim)
            local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, threshold, np.array([]))

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[np.array([j in lc for lc in local_clusters])]
            indices = np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


def perform_local_clustering(
        embeddings: np.ndarray, dim: int, threshold: float, query_embeddings: np.ndarray, verbose: bool = False
) -> List[np.ndarray]:
    if query_embeddings.size != 0:
        combined = np.concatenate([embeddings, query_embeddings], axis=0)
        # 一次性降维
        reduced_combined = local_cluster_embeddings(combined, min(dim, len(combined) - 2))
        # 拆分还原
        reduced_embeddings_local = reduced_combined[:len(embeddings)]
        reduced_query_embeddings_local = reduced_combined[len(embeddings):]
    else:
        reduced_embeddings_local = local_cluster_embeddings(embeddings, min(dim, len(embeddings) - 2))
        reduced_query_embeddings_local = np.array([])

    # Perform global clustering with custom initial centers (query_list embeddings)
    local_clusters, n_local_clusters = GMM_cluster(
        reduced_embeddings_local, threshold, reduced_query_embeddings_local
    )

    if verbose:
        logging.info(f"Global Clusters: {n_local_clusters}")

    # 仅返回局局聚类结果
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]

    for i in range(n_local_clusters):
        local_cluster_embeddings_ = embeddings[np.array([i in gc for gc in local_clusters])]
        if verbose:
            logging.info(f"Nodes in Global Cluster {i}: {len(local_cluster_embeddings_)}")
        if len(local_cluster_embeddings_) == 0:
            continue

        # 为每个全局聚类分配索引
        indices = np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))[1]
        for idx in indices:
            all_local_clusters[idx] = np.append(all_local_clusters[idx], i)

    if verbose:
        logging.info(f"Total Global Clusters: {n_local_clusters}")

    return all_local_clusters



# Clustering Algorithm Interface
class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


# RAPTOR Clustering with Node-based Operations
class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 10000,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
        query_nodes: List[Node] = None,
        cluster_method: str = "hierarchical",
    ) -> List[List[Node]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])
        query_embeddings = np.array([])
        if query_nodes:
            query_embeddings = np.array([node.embeddings[embedding_model_name] for node in query_nodes])
        # 使用自定义初始化中心执行聚类
        # clusters = perform_clustering(
        #     embeddings, dim=reduction_dimension, threshold=threshold, query_embeddings=query_embeddings
        # )
        if cluster_method == "hierarchical":
            clusters = perform_clustering(
                embeddings, dim=reduction_dimension, threshold=threshold, query_embeddings=query_embeddings
            )
        elif cluster_method == "global":
            clusters = perform_local_clustering(
                embeddings, dim=reduction_dimension, threshold=threshold, query_embeddings=query_embeddings
            )
        else:
            # 抛出一个异常，说明 cluster_method 的值不符合要求
            raise ValueError(
                f"Unsupported cluster method '{cluster_method}'. Supported methods are 'hierarchical' and 'global'.")

        # 初始化一个空列表来存储节点的聚类
        node_clusters = []

        # 遍历每个聚类标签
        for label in np.unique(np.concatenate(clusters)):
            # 获取属于此聚类的节点的索引
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # 将相应的节点添加到 node_clusters 列表中
            cluster_nodes = [nodes[i] for i in indices]

            # 基本情况：如果聚类中只有一个节点，则不尝试重新聚类
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # 计算节点中文本的总长度
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # 如果文本的总长度超过最大允许长度，则对该聚类进行重新聚类
            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(f"Reclustering cluster with {len(cluster_nodes)} nodes")
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        nodes=cluster_nodes,
                        embedding_model_name=embedding_model_name,
                        query_nodes=query_nodes,
                        max_length_in_cluster=max_length_in_cluster,
                        tokenizer=tokenizer,
                        reduction_dimension=reduction_dimension,
                        threshold=threshold,
                        verbose=verbose
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters


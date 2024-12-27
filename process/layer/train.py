# train.py

import os
import torch
from torch.utils.data import DataLoader
from model import UNetAutoencoder  # 确保您有定义的 UNetAutoencoder 模型
from dataset import load_datasets  # 确保您有定义的 load_datasets 函数
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, f1_score, average_precision_score
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy
import joblib
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 可视化


def train_and_validate_model(train_dataset, test_dataset, learning_rate=0.001, batch_size=32, num_epochs=100,
                             patience=10, output_dir='output'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetAutoencoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    train_losses = []
    val_losses = []
    lr_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, _ = model(data)  # 解码器输出
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                output, _ = model(data)  # 解码器输出
                loss = criterion(output, data)
                total_test_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_test_loss)

        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_test_loss:.6f}, Learning Rate: {current_lr:.6f}')

        scheduler.step()

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_model_wts)

    # 绘制并保存训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_validation_loss_curve.png'), dpi=300)
    plt.close()

    # 绘制学习率调度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(lr_list) + 1), lr_list, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_schedule.png'), dpi=300)
    plt.close()

    return model, train_losses, val_losses, lr_list


def extract_features(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    features = []
    filenames = []
    with torch.no_grad():
        for data, filename in data_loader:
            data = data.to(device)
            _, enc4 = model(data)
            features.append(enc4.view(enc4.size(0), -1).cpu().numpy())
            # 将 filename 转换为字符串
            if isinstance(filename, torch.Tensor):
                try:
                    # 假设 filename 是包含字节的单元素张量
                    fname_str = ''.join([chr(c) for c in filename.cpu().numpy().flatten() if c != 0])
                    filenames.append(fname_str)
                except:
                    # 回退方案
                    fname_str = str(filename)
                    filenames.append(fname_str)
            elif isinstance(filename, (list, tuple)):
                if len(filename) > 0:
                    fname = filename[0]
                    if isinstance(fname, torch.Tensor):
                        try:
                            fname_str = ''.join([chr(c) for c in fname.cpu().numpy().flatten() if c != 0])
                            filenames.append(fname_str)
                        except:
                            fname_str = str(fname)
                            filenames.append(fname_str)
                    else:
                        filenames.append(str(fname))
                else:
                    filenames.append("unknown")
            else:
                filenames.append(str(filename))
    return np.concatenate(features), filenames


def perform_clustering(features, method='kmeans', **kwargs):
    method = method.lower()
    if method == 'kmeans':
        clustering_model = KMeans(**kwargs).fit(features)
    elif method == 'gmm':
        clustering_model = GaussianMixture(**kwargs).fit(features)
    elif method == 'agglomerative':
        clustering_model = AgglomerativeClustering(**kwargs).fit(features)
    elif method == 'dbscan':
        clustering_model = DBSCAN(**kwargs).fit(features)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    return clustering_model


def save_cluster_results(filenames, labels, output_path):
    clusters = {}
    for filename, label in zip(filenames, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(filename)

    with open(output_path, 'w') as f:
        for label, files in clusters.items():
            f.write(f"Cluster {label}:\n")
            for file in files:
                f.write(f"  {file}\n")
            f.write("\n")


def visualize_clusters(features, labels, method_name, output_dir):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.6,
                          edgecolors='w')
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization using {method_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method_name}_cluster_visualization.png'), dpi=300)
    plt.close()


def visualize_clusters_3D(features, labels, method_name, output_dir):
    # 使用 PCA 进行 3D 降维
    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2],
                         c=labels, cmap='viridis', alpha=0.6, edgecolors='w')

    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
    ax.add_artist(legend1)

    ax.set_title(f'3D Cluster Visualization using {method_name}')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{method_name}_cluster_visualization_3D.png'), dpi=300)
    plt.close()


def evaluate_clustering(labels, features):
    if len(set(labels)) <= 1:
        print("Only one cluster found, skipping silhouette score.")
        return -1
    silhouette_avg = silhouette_score(features, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    return silhouette_avg


def extract_raw_features(dataset):
    raw_features = []
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    for data, _ in data_loader:
        matrix = data.squeeze(0).squeeze(0).numpy()
        raw_features.append(matrix.flatten())
    return np.array(raw_features)


def plot_silhouette_comparison(raw_score, deep_scores, output_dir):
    methods = list(deep_scores.keys())
    scores = list(deep_scores.values())

    plt.figure(figsize=(10, 6))
    plt.bar(methods, scores, color='skyblue')
    plt.axhline(y=raw_score, color='red', linestyle='--', label='Raw Features Silhouette Score')
    plt.xlabel('Clustering Method')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Clustering Methods on Deep Features')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_methods_silhouette_scores.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    labels_plot = ['Raw Features'] + methods
    comparison_scores = [raw_score] + scores
    colors = ['orange'] + ['skyblue'] * len(methods)
    plt.bar(labels_plot, comparison_scores, color=colors)
    plt.xlabel('Feature Type / Clustering Method')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Comparison between Raw and Deep Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'silhouette_score_comparison_raw_deep.png'), dpi=300)
    plt.close()


def plot_cluster_evaluation(comparison_scores, output_dir):
    methods = list(comparison_scores.keys())
    scores = list(comparison_scores.values())

    plt.figure(figsize=(10, 6))
    plt.bar(methods, scores, color='lightgreen')
    plt.xlabel('Clustering Method')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different Clustering Methods')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_evaluation_comparison.png'), dpi=300)
    plt.close()


def compute_multilabel_metrics(pred_labels, true_labels):
    """
    计算多标签的F1分数和AUPR。

    参数：
    - pred_labels: 预测标签，形状为 (n_samples, n_labels)
    - true_labels: 真实标签，形状为 (n_samples, n_labels)

    返回：
    - micro_f1: 微平均F1分数
    - macro_f1: 宏平均F1分数
    - micro_aupr: 微平均AUPR
    - macro_aupr: 宏平均AUPR
    """
    # F1分数
    micro_f1 = f1_score(true_labels, pred_labels, average='micro', zero_division=0)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    # AUPR
    try:
        micro_aupr = average_precision_score(true_labels, pred_labels, average='micro')
    except ValueError:
        micro_aupr = 0.0
    try:
        macro_aupr = average_precision_score(true_labels, pred_labels, average='macro')
    except ValueError:
        macro_aupr = 0.0

    return micro_f1, macro_f1, micro_aupr, macro_aupr


def load_true_labels_from_tsv(tsv_file, target_iprs):
    """
    加载多标签TSV文件，返回一个DataFrame。

    参数：
    - tsv_file: 标签TSV文件路径。
    - target_iprs: 目标IPR编号列表。

    返回：
    - df: 包含seq_id和目标IPR列的DataFrame。
    """
    if not os.path.exists(tsv_file):
        print(f"TSV file '{tsv_file}' not found. No labels loaded.")
        return None

    df = pd.read_csv(tsv_file, sep='\t')

    # 确保 'seq_id' 存在
    if 'seq_id' not in df.columns:
        print("TSV 文件中缺少 'seq_id' 列。")
        return None

    # 检查并添加缺失的 IPR 列，填充为 0
    for ipr in target_iprs:
        if ipr not in df.columns:
            print(f"Warning: {ipr} not found in TSV file columns.")
            df[ipr] = 0  # 填充缺失列为 0

    return df[['seq_id'] + target_iprs]


def map_labels_to_clusters(true_labels, labels, num_clusters):
    """
    将每个标签分配给在该标签上最频繁的聚类。

    参数：
    - true_labels: numpy 数组，形状为 (n_samples, n_labels)
    - labels: numpy 数组，形状为 (n_samples,)
    - num_clusters: 聚类数量

    返回：
    - cluster_to_labels: 字典，映射聚类索引到标签索引的列表
    """
    cluster_to_labels = {i: [] for i in range(num_clusters)}
    n_labels = true_labels.shape[1]
    for label_idx in range(n_labels):
        # 找到该标签在各个聚类中的频率
        label_counts = []
        for cluster in range(num_clusters):
            indices = np.where(labels == cluster)[0]
            if len(indices) == 0:
                count = 0
            else:
                count = np.sum(true_labels[indices, label_idx])
            label_counts.append(count)
        best_cluster = np.argmax(label_counts)
        if label_counts[best_cluster] > 0:
            cluster_to_labels[best_cluster].append(label_idx)
    return cluster_to_labels


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"  # 避免在 Windows 上的内存泄漏

    # 定义输出目录
    output_dir = ''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载数据集
    train_dataset, test_dataset, full_dataset = load_datasets(
        '',
        target_size=(256, 256))

    # 训练模型
    trained_model, train_losses, val_losses, lr_list = train_and_validate_model(
        train_dataset, test_dataset, learning_rate=0.001, batch_size=32,
        num_epochs=100, patience=10, output_dir=output_dir)

    # 保存训练好的模型
    torch.save(trained_model.state_dict(), os.path.join(output_dir, 'trained_model.pth'))

    # 提取深度特征
    deep_features, filenames = extract_features(trained_model, full_dataset)

    # 加载多标签TSV文件（仅包含前50个IPR）
    # 请确保您已经运行了 convert_tsv_to_labels.py 并生成了 output_labels_top50.tsv
    multi_label_tsv = ""

    # 动态读取目标IPR编号列表（前50个）
    if os.path.exists(multi_label_tsv):
        df_top_labels = pd.read_csv(multi_label_tsv, sep='\t')
        target_iprs = [col for col in df_top_labels.columns if col != 'seq_id']
    else:
        print(f"Multi-label TSV file '{multi_label_tsv}' does not exist.")
        target_iprs = []

    # 加载真实标签
    df_labels = load_true_labels_from_tsv(multi_label_tsv, target_iprs)
    if df_labels is not None:
        print(f"Loaded true labels from {multi_label_tsv}.")
    else:
        print("No labels loaded. F1 和 AUPR 将被跳过。")

    # 根据 filenames 顺序构建 true_labels 矩阵
    true_labels = None
    if df_labels is not None:
        # 创建一个从 seq_id 到标签的字典
        seq_to_labels = df_labels.set_index('seq_id').to_dict(orient='index')
        # 初始化 true_labels
        true_labels = []
        for fname in filenames:
            # 从 filename 中提取 seq_id
            seq_id = os.path.splitext(os.path.basename(fname))[0]
            labels = seq_to_labels.get(seq_id, {ipr: 0 for ipr in target_iprs})
            label_list = [labels.get(ipr, 0) for ipr in target_iprs]
            true_labels.append(label_list)
        true_labels = np.array(true_labels)
        print(f"Constructed true_labels matrix with shape: {true_labels.shape}")
        print("Sum of each label in true_labels:")
        print(true_labels.sum(axis=0))
    else:
        print("No usable labels found from TSV.")

    # 检查是否存在至少一个 '1'
    if true_labels is not None and np.sum(true_labels) == 0:
        print("All labels are 0. 请检查标签文件确保包含一些 '1'。")
        exit(1)

    # 执行 PCA 进行降维（用于聚类和可视化）
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(deep_features)

    # 定义聚类方法及其参数
    clustering_methods = {
        'KMeans': {'n_clusters': 5, 'random_state': 0, 'n_init': 10},
        'GMM': {'n_components': 5, 'random_state': 0},
        'Agglomerative': {'n_clusters': 5},
        'DBSCAN': {'eps': 0.5, 'min_samples': 5}
    }

    deep_silhouette_scores = {}
    best_score = -1
    best_method = None
    best_labels = None

    for method, params in clustering_methods.items():
        clustering_model = perform_clustering(reduced_features, method=method, **params)

        if method.lower() == 'dbscan':
            labels = clustering_model.labels_
        elif method.lower() == 'gmm':
            labels = clustering_model.predict(reduced_features)
        else:
            labels = clustering_model.labels_

        score = evaluate_clustering(labels, reduced_features)
        deep_silhouette_scores[method] = score

        # 如果有真实标签则计算多标签F1和AUPR
        if true_labels is not None and len(true_labels) == len(labels):
            unique_clusters = set(labels)
            num_clusters = len(unique_clusters)
            cluster_to_labels = map_labels_to_clusters(true_labels, labels, num_clusters)
            print(f"Cluster to labels mapping for {method}: {cluster_to_labels}")

            # 根据聚类结果生成预测标签
            pred_multilabel = np.zeros_like(true_labels)
            for i, cluster in enumerate(labels):
                assigned_labels = cluster_to_labels.get(cluster, [])
                for label_idx in assigned_labels:
                    pred_multilabel[i, label_idx] = 1

            # 计算多标签F1和AUPR
            micro_f1, macro_f1, micro_aupr, macro_aupr = compute_multilabel_metrics(pred_multilabel, true_labels)
            print(f"Method: {method}, Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}")
            print(f"Method: {method}, Micro AUPR: {micro_aupr:.4f}, Macro AUPR: {macro_aupr:.4f}")

        if score > best_score:
            best_score = score
            best_method = method
            best_labels = labels

        # 保存聚类模型
        joblib.dump(clustering_model, os.path.join(output_dir, f'{method}_model.pkl'))

        # 可视化并保存聚类结果
        try:
            visualize_clusters_3D(deep_features, labels, method, output_dir)
        except ValueError:
            visualize_clusters(reduced_features, labels, method, output_dir)

    print(f"Best method: {best_method}, Best Silhouette Score: {best_score}")

    # 保存最佳聚类结果
    if best_labels is not None:
        save_cluster_results(filenames, best_labels, os.path.join(output_dir, f'cluster_results_{best_method}.txt'))

    # 对比原始特征和深度特征的聚类
    raw_features = extract_raw_features(full_dataset)
    pca_raw = PCA(n_components=2)
    reduced_raw_features = pca_raw.fit_transform(raw_features)
    kmeans_raw = KMeans(n_clusters=5, random_state=0, n_init=10).fit(reduced_raw_features)
    labels_raw = kmeans_raw.labels_

    raw_silhouette = evaluate_clustering(labels_raw, reduced_raw_features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_raw_features[:, 0], reduced_raw_features[:, 1], c=labels_raw, cmap='viridis',
                          alpha=0.6, edgecolors='w')
    plt.colorbar(scatter)
    plt.title('Cluster Visualization using Raw Features and K-Means')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kmeans_raw_cluster_visualization.png'), dpi=300)
    plt.close()

    plot_silhouette_comparison(raw_silhouette, deep_silhouette_scores, output_dir)
    plot_cluster_evaluation(deep_silhouette_scores, output_dir)

    print("All visualization plots have been generated and saved to the output directory.")

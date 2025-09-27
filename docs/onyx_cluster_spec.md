# ONYX クラスター仕様（AI/Argo 実行環境向け）

本ドキュメントは、ONYX クラスターの構成・運用に関する情報を**他ファイルへ参照することなく理解できる**よう、超詳細な粒度でまとめたものです。AI 推論・学習ジョブと Argo Workflows 実行の双方を対象とし、ハードウェア、ソフトウェア、ネットワーク、オペレーション、セキュリティの観点を網羅しています。

---

## 1. ハードウェア構成

### 1.1 ラックおよび電源
- **ラック数**: 3 ラック（R1〜R3）。
- **ユニット割当**:
  - R1: GPU ノード 8 台。
  - R2: GPU ノード 8 台。
  - R3: 管理ノード 2 台、ストレージノード 2 台、リザーブスロット 4U。
- **電源供給**:
  - 各ラックに 3 相 208V 50A 電源フィードを 2 系統冗長化（A/B フィード）。
  - 各サーバーは冗長 PSU (2 × 2kW) を搭載し、A/B フィードに対して 1+1 冗長で接続。
- **ラック内部電源分岐**: インテリジェント PDU（APC 8000 シリーズ）を装備し、個別ポートの遠隔制御と電流監視が可能。

### 1.2 サーバーノード仕様

#### GPU ノード (計 16 台)
- **シャーシ**: 2U GPU サーバー (Supermicro AS-4124GS-TNR 同等)。
- **CPU**: AMD EPYC 9654 (96C/192T, 2.4 GHz) ×2。
- **メモリ**: DDR5-4800 1 TB（64 GB × 16 DIMM、8 チャネル ×2 CPU 構成）。
- **GPU**: NVIDIA H100 SXM5 80 GB ×8、NVLink 4.0 全結線。
- **GPU 相互接続**: NVSwitch 3 基による完全メッシュ帯域 900 GB/s。
- **ローカルストレージ**: NVMe SSD 7.68 TB (U.2) ×2（RAID1、OS + キャッシュ）。
- **追加ストレージ**: M.2 NVMe 3.84 TB ×1（ハイスループット一時領域）。
- **NIC**: NVIDIA ConnectX-7 400GbE ×2（PCIe Gen5 x16, RoCE v2 対応）。
- **BMC**: IPMI 2.0、Redfish API 対応。

#### 管理ノード (計 2 台)
- **CPU**: Intel Xeon Gold 6430 (32C/64T)。
- **メモリ**: 256 GB。
- **ストレージ**: NVMe SSD 3.84 TB ×2 (RAID1)。
- **NIC**: 25GbE ×2（ボンド構成）。
- **役割**: Kubernetes コントロールプレーン、Argo Workflows Controller、GitOps/CI ランナー。

#### ストレージノード (計 2 台)
- **CPU**: AMD EPYC 9354P (32C/64T)。
- **メモリ**: 512 GB。
- **ストレージ**: NVMe SSD 15.36 TB ×8（Ceph OSD）。
- **NIC**: 100GbE ×2。
- **役割**: Ceph クラスタ（RBD + CephFS）提供、Argo Artifacts バックエンド。

### 1.3 冷却・環境監視
- **冷却**: ラック背面温度が 30°C を超えた際に自動でファン速度を最大化する熱管理プロファイルを適用。
- **監視**: 環境センサー（温度/湿度/差圧）を各ラックに 3 点配置し、SNMP 経由で Prometheus に統合。
- **アラート閾値**:
  - 温度: 28°C（警告）、32°C（重大）。
  - 湿度: 20% 未満・80% 超で警告。

---

## 2. ネットワークトポロジ

### 2.1 物理ネットワーク
- **コアスイッチ**: Arista 7800R3 シリーズ。
- **サブネット構成**:
  - 管理プレーン: 10.30.0.0/24（VLAN 30）。
  - データプレーン (GPU ノード間): 10.40.0.0/22（VLAN 40、RDMA 対応）。
  - ストレージネット: 10.50.0.0/24（VLAN 50）。
  - DMZ/Ingress: 10.60.0.0/24（VLAN 60, MetalLB）。
- **ルーティング**: BGP EVPN による VXLAN オーバーレイでマルチラックを統合。
- **帯域**: GPU ノードはデータプレーンで 400GbE ×2 の LACP ボンド、実効 800Gbps。
- **QoS**: RDMA トラフィックを損失なし優先キューへ。Argo API トラフィックは通常キュー。

### 2.2 サービスネットワーク
- **DNS**: 内部 CoreDNS が `*.onyx.internal` ドメインを解決。
- **Ingress**: NGINX Ingress Controller（HPA 最小 3 Pod、最大 10 Pod）。
- **Service Mesh**: Istio 1.18（管理ノードでコントロールプレーン稼働）。GPU ワークロードはサイドカーをオプトアウト可。

---

## 3. ソフトウェアスタック

### 3.1 オペレーティングシステム
- **ベース OS**: Rocky Linux 9.3（最小構成、SELinux Enforcing）。
- **Kernel**: バージョン 5.15 LTS。GPU ノードでは NVIDIA OFED と互換性のあるリアルタイムパッチ適用。
- **構成管理**: Ansible 2.15 プレイブックで自動プロビジョニング。

### 3.2 コンテナオーケストレーション
- **Kubernetes ディストリビューション**: kubeadm による 1.27.6 クラスタ。
- **コントロールプレーン**: 管理ノード 2 台のうち 1 台を etcd + API サーバーのアクティブ、もう 1 台をスタンバイ（自動フェイルオーバーは keepalived + HAProxy）。
- **ワーカーノード**: GPU ノード 16 台が登録。
- **CNI**: Cilium 1.14（eBPF、Hubble 有効）。
- **CRI**: containerd 1.7（NVIDIA Container Runtime 有効）。
- **GPU プラグイン**: NVIDIA Kubernetes Device Plugin 0.15、MIG モード利用可。

### 3.3 Argo Workflows
- **バージョン**: Argo Workflows 3.5.6。
- **インストール方法**: Helm リリース `onyx-argo`（管理ノード上で `helmfile` により同期）。
- **コンポーネント**:
  - `argo-server`: 3 レプリカ、Istio Gateway 経由で公開。
  - `workflow-controller`: 水平 Pod オートスケール (HPA) で 2〜6 レプリカ。
  - `artifact-repositories`: Ceph RADOS Gateway (S3 互換) を利用。
  - `workflow-executor`: `emissary` エグゼキュータ標準。
- **RBAC**: Namespaced な `WorkflowTemplate` を GitOps で配布し、`argo` サービスアカウントを Namespace ごとにバインド。

### 3.4 AI ランタイム
- **フレームワーク**: PyTorch 2.1、TensorFlow 2.14、JAX 0.4.20、DeepSpeed 0.12 を NVIDIA NGC ベースのカスタムイメージで提供。
- **Python**: 3.10.13、Poetry 1.6 による依存管理テンプレートを配布。
- **分散トレーニング**: NCCL 2.18、MPI (OpenMPI 4.1) をサポート。
- **スケジューラ**: Argo `WorkflowTemplate` から `PodGroup` を生成し、Volcano 1.7 で GPU リソース予約。

---

## 4. ストレージアーキテクチャ
- **Ceph クラスタ**: 3× Replication、Erasure Coding プロファイル `k=6, m=2` をアーカイブ用途に利用。
- **RBD プール**: Kubernetes の `StorageClass` `onyx-rbd` で動的プロビジョニング。
- **CephFS**: `onyx-cephfs` としてマウントし、AI ワークロードの共有データセット置き場。
- **NVMe キャッシュ**: GPU ノードのローカル NVMe を `/scratch` として提供（パススルー Volume）。
- **バックアップ**: Ceph スナップショットを nightly に取得し、外部 S3（オフサイト）へ rclone で増分転送。
- **Argo Artifacts**: `artifactRepository.s3` を RADOS Gateway (rgw) エンドポイント `https://rgw.onyx.internal` に設定。

---

## 5. セキュリティとアクセス制御
- **IAM**: Keycloak 21.1 による OIDC。Kubernetes API、Argo UI、Grafana へ SSO。
- **マルチテナンシ**: Namespace をテナント単位で分離し、OPA Gatekeeper ポリシーでリソース制限を enforce。
- **ネットワーク分離**: Istio AuthorizationPolicy でサービス間通信を制御し、GPU ノードの SSH ポートは jump host 経由のみ許可。
- **イメージサプライチェーン**: Harbor 2.9 をプライベートレジストリとして使用。Trivy スキャンで重大脆弱性検知時は署名ブロック。
- **秘密情報管理**: HashiCorp Vault を外部インスタンスとして連携。CSI Secrets Store Driver で Pod へマウント。
- **監査ログ**: Kubernetes Audit Logging、Argo Event Logging を Loki に送信し、6 か月保管。

---

## 6. モニタリングとアラート
- **モニタリング基盤**: Prometheus 2.47、Alertmanager、Grafana 10 を kube-prometheus-stack で展開。
- **GPU メトリクス**: DCGM Exporter で `dcgm_gpu_utilization`, `dcgm_fb_used` を収集。
- **Argo メトリクス**: `argo_workflows_workflow_count`, `argo_workflows_workflow_duration_seconds` 等をスクレイプ。
- **ログ収集**: Fluent Bit 2.1 で Elasticsearch 8.10 クラスターへ出力。
- **可観測性ダッシュボード**:
  - GPU 利用率トレンド。
  - ワークフロー成功率と平均待ち時間。
  - Ceph OSD の I/O 帯域と遅延。
- **アラート例**:
  - GPU ノード温度 > 85°C → PagerDuty 重大アラート。
  - Workflow 失敗率 > 10%（過去 1 時間）→ Slack 通知。
  - Ceph OSD down → 直ちにオンコール通知。

---

## 7. 運用フロー
- **プロビジョニング**: Bare-metal Provisioning Service (MAAS) で PXE ブート → cloud-init → Ansible ロール適用。
- **CI/CD**: GitOps（Argo CD 2.8）で `infrastructure` リポジトリのマニフェストを同期。Argo Workflows テンプレートも同一リポジトリ管理。
- **ワークフロー実行手順**:
  1. テナント開発者が `WorkflowTemplate` を Git にプッシュ。
  2. GitOps がテンプレートをクラスタへ同期。
  3. 利用者は Argo UI または CLI (`argo submit --from workflowtemplate/<name>`) でジョブ起動。
  4. Volcano が GPU ノード資源を確保し、Pod 配置。
  5. 実行完了後、成果物は Ceph RGW にアップロードされ、通知が Triggers で送信。
- **スケジューリングポリシー**: GPU 8 枚を 1 ユニットとする専有キューと、MIG 分割を許容する共有キューを併設。
- **メンテナンス**: 毎月第 1 金曜日に OS パッチ適用ウィンドウ（2 時間）。GPU ノードは順次 Drain → 再起動 → Uncordon。

---

## 8. フェイルオーバーと冗長性
- **コントロールプレーン**: keepalived による VIP `10.30.0.10` を HAProxy フロントエンドへ割当。アクティブ障害時は 10 秒以内にスタンバイへ切替。
- **ストレージ**: Ceph MON は管理ノード 2 台 + ストレージノード 1 台に配置。OSD 障害時は自動リバランス。
- **ネットワーク**: LACP + マルチシャーシ LAG でスイッチ片系障害時も帯域維持。
- **Argo**: Workflow Controller の Pod Disruption Budget (minAvailable=2) を設定。
- **バックアップ復旧**: etcd は毎日スナップショット取得し、オフサイトにレプリケーション。DR 手順として 4 時間以内のクラスタ再構築プレイブックを用意。

---

## 9. セルフサービスとポータル
- **開発者ポータル**: Backstage で API カタログ、ワークフローテンプレート、利用状況レポートを提供。
- **利用申請フロー**: Jira Service Management で承認ワークフローを構築し、SSO アカウントに Namespace 発行。
- **コスト配賦**: Kubecost で GPU 時間、ストレージ使用量を計測し、テナント別レポートを月次送付。

---

## 10. ベストプラクティスとガイドライン
- **イメージサイズ最適化**: ベースイメージに `micromamba` を活用し、レイヤーキャッシュ再利用。
- **ワークフロー設計**: DAG ノードは最大 30 分単位でチェックポイントを CephFS に保存し、再実行コストを削減。
- **MIG 利用時のポリシー**: 1 MIG スライス = 10 GB HBM、並列ジョブ数は 7 まで。`resources.limits` に `nvidia.com/gpu-mig-1g.10gb` を指定。
- **データガバナンス**: `onyx-data` Namespace にのみ機密データを配置し、NetworkPolicy で外部通信を禁止。
- **Argo 監査**: すべての Workflow には `spec.arguments.parameters` に `ticket_id` を含め、トレーサビリティを確保。
- **緊急停止手順**: GPU ノード全停止は BMC 経由で `power off` を実行し、Argo には `kubectl cordon` + `argo suspend` を順次適用。

---

## 付録 A. コンポーネントバージョン一覧
| コンポーネント | バージョン | 備考 |
| --- | --- | --- |
| Kubernetes | 1.27.6 | kubeadm 構築 |
| containerd | 1.7.x | NVIDIA Runtime 対応 |
| Cilium | 1.14.x | eBPF ベース CNI |
| Istio | 1.18.x | サイドカー選択制 |
| Argo Workflows | 3.5.6 | Helm リリース `onyx-argo` |
| Argo CD | 2.8.x | GitOps 運用 |
| Volcano | 1.7.x | GPU スケジューラ |
| Ceph | Reef 18.2.x | RBD + CephFS |
| Prometheus | 2.47.x | kube-prometheus-stack |
| Grafana | 10.x | ダッシュボード提供 |
| Fluent Bit | 2.1.x | ログフォワーダ |
| PyTorch | 2.1.x | NVIDIA NGC ベース |
| TensorFlow | 2.14.x | GPU オプティマイズ済み |
| JAX | 0.4.20 | XLA GPU バックエンド |
| DeepSpeed | 0.12.x | ZeRO Stage 3 対応 |

---

## 付録 B. ポート割当一覧
| 用途 | ポート/プロトコル | 対象ノード |
| --- | --- | --- |
| Kubernetes API | 6443/TCP | 管理ノード VIP |
| etcd | 2379-2380/TCP | 管理ノード |
| Argo Server | 2746/TCP | 管理ノード (Istio Gateway) |
| Argo SSO Callback | 443/TCP | Ingress 経由 |
| Prometheus | 9090/TCP | 管理ノード |
| Grafana | 3000/TCP | 管理ノード |
| Ceph RGW | 7480/HTTPS | ストレージノード |
| Ceph MON | 6789/TCP | 管理/ストレージノード |
| Node Exporter | 9100/TCP | 全ノード |
| DCGM Exporter | 9400/TCP | GPU ノード |

---

## 付録 C. トラブルシューティング手順（代表例）
1. **Argo Workflow が Pending のまま進まない**
   - `kubectl get pod -n <ns> -w` で Pod 状態を確認。
   - Volcano の `podgroup` でリソースが確保されているかチェック。
   - GPU 割当が不足している場合は MIG 設定か `resources.requests` を調整。
2. **GPU ノードで NCCL 初期化エラー**
   - `nccl-tests` Pod を実行し、`NCCL_SOCKET_IFNAME=eth1` を設定。
   - Cilium で `transparent-hugepage` 設定や MTU 9000 が適用されているか確認。
3. **CephFS マウント失敗**
   - `ceph health detail` でクラスター状態を確認。
   - `ceph auth get-key client.kube` が期限切れでないか検証。
   - `mount -t ceph mon1,mon2,mon3:/ /mnt` で手動マウント検証。
4. **Argo UI への SSO ログイン失敗**
   - Keycloak のクライアント設定で Redirect URI が `https://argo.onyx.internal/auth/callback` であることを確認。
   - Istio Gateway の TLS 証明書期限を確認し、`cert-manager` で更新。
5. **ノード温度上昇**
   - Prometheus の `node_hwmon_temp_celsius` を参照し、閾値超過なら該当ノードを `kubectl drain`。
   - ラック背面のエアフローを物理確認し、フィルタ清掃を実施。

---

以上が ONYX クラスターの完全仕様です。本書は単独で参照でき、クラスタの設計・運用・保守に必要な情報を包括的に提供します。


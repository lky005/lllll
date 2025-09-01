# 网络 QoS（QCI/5QI）初步分析报告

- 采集包：LOG_20250731144213
- 目标：定位是否出现蜂窝网络 QCI/5QI/QFI 或相关策略（AMBR/GBR/ARP/DRB/ERAB/SNSSAI）

## 结论摘要
- 目前未发现与蜂窝 QCI/5QI 直接相关的命中；已见到的 qos[n] 为应用线程调度日志（ffrt）。

## 关键命中（示例）
- 请将 content_hits*.txt / hilog_*_hits.txt / sysevent_hits.txt 中的代表性命中粘贴在此。

## 缺失与建议
- 包内未包含 telephony/cellular/modem/ril/diag/pcap 等基带层日志。
- 建议：开启 Telephony/Cellular/NetManager 详细日志或厂商 modem/diag 抓取。

## 附件与清单
- results_bundle.zip（汇总）
- name_hits.txt
- content_hits.txt / content_hits_context.txt
- hilog_unzipped_hits.txt / hilog_diag_hits.txt
- sysevent_hits.txt
- cmd_output_hits.txt
- top_files.txt
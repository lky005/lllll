# QCI/5QI 扫描结果（LOG_20250731144213）

内容：
- results_bundle.zip：聚合的命中与清单
- name_hits.txt / content_hits*.txt / hilog_*_hits.txt / sysevent_hits.txt / cmd_output_hits.txt
- top_files.txt（体积排行）

如何复现本地扫描：
1. 在日志目录运行 qci_qos_scanner.py：
   
   python .\qci_qos_scanner.py "C:\\path\\to\\LOG_20250731144213"
2. 生成的 txt 放入本目录，或压缩为 results_bundle.zip

备注：
- 命中中的 "ffrt qos[n]" 属于应用线程调度，非蜂窝网络 QCI/5QI。
- 若需获取基带/蜂窝层日志，请开启 Telephony/Cellular 详细日志或厂商 diag 采集。
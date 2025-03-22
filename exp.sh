#!/bin/bash

# 指定要运行脚本的目录
directory="scripts/csi300"

# 指定日志目录
log_directory="log"

# 检查脚本目录是否存在
if [ ! -d "$directory" ]; then
  echo "错误: 目录 '$directory' 不存在。"
  exit 1
fi

# 创建日志目录（如果不存在）
mkdir -p "$log_directory"

# 循环遍历目录中的所有 sh 文件
find "$directory" -maxdepth 1 -name "*.sh" -print0 | while IFS= read -r -d $'\0' script; do
  # 获取脚本文件名（不带路径和扩展名）
  script_name=$(basename "$script" .sh)

  # 构建日志文件名
  log_file="$log_directory/${script_name}.log"

  echo "正在运行脚本: '$script'，日志将保存到 '$log_file'..."

  # 运行脚本并将输出重定向到日志文件
  if bash "$script" &> "$log_file"; then
    echo "脚本 '$script' 成功运行。"
  else
    echo "脚本 '$script' 运行失败。请检查 '$log_file' 以获取更多信息。"
  fi
done

echo "完成运行所有脚本。"


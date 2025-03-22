import os
import glob

def combine_logs(output_file):
    """
    将当前目录下所有 .log 文件的内容合并到一个 .txt 文件中。

    Args:
        output_file: 输出的 .txt 文件路径。
    """

    log_dir = "."  #  当前目录

    log_files = glob.glob(os.path.join(log_dir, "*.log"))

    if not log_files:
        print(f"在当前目录中没有找到 .log 文件。")
        return

    try:
        with open(output_file, "w", encoding="utf-8") as outfile:
            for log_file in log_files:
                with open(log_file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
                outfile.write("\n")  # 在每个文件内容后添加换行符，以区分不同文件

        print(f"已将当前目录下所有 .log 文件的内容合并到 {output_file}")

    except Exception as e:
        print(f"发生错误：{e}")

# 使用示例：
output_txt_file = "combined_logs.txt"  # 输出的 txt 文件名

combine_logs(output_txt_file)

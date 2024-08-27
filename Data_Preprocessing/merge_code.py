import os
import re

output_file = "merged_code.txt"  # 合并后的输出文件名
base_directory = "./"  # 包含所有contract文件夹的基础目录

total_folders = 41  # 文件夹总数
processed_folders = 0  # 已处理的文件夹计数

with open(output_file, "w") as outfile:
    for folder in range(1, total_folders + 1):  # 遍历41个文件夹
        folder_name = "contract" + str(folder)
        folder_path = os.path.join(base_directory, folder_name)
        if not os.path.exists(folder_path):
            continue

        processed_folders += 1
        print(f"正在处理文件夹 {folder_name} ({processed_folders}/{total_folders})")

        for file_name in os.listdir(folder_path):  # 遍历每个文件夹中的txt文件
            if not file_name.endswith(".txt"):
                continue

            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as infile:
                code = infile.read()
                # 删除 /* */ 类型的注释
                code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
                # 删除 // 类型的注释
                code = re.sub(r"\/\/.*", "", code)
                # 删除注释符后面的内容
                code = re.sub(r"\/\*.*|\/\/.*", "", code)
                # 合并代码行
                code = re.sub(r"\n", " ", code)
                # 在符号括号和字母单词之间添加一个空格
                code = re.sub(r"([^\s\w]|(?<=\w)(?=[^\s\w]))", r" \1 ", code)
                # 删除多余的空格并且保持单词间一个空格的距离
                code = re.sub(r"\s+", " ", code)
                outfile.write(code + "\n")

print("合并完成！")

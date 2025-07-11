#!/usr/bin/env python3
"""
清理requirements-frozen.txt，处理file://依赖
"""
import re

# 读取requirements-frozen.txt
try:
    with open('requirements-frozen.txt', 'r') as f:
        lines = f.readlines()
except FileNotFoundError:
    print("错误：找不到 requirements-frozen.txt 文件")
    print("请先执行: docker cp 7ee9b6b80c0e:/tmp/requirements-frozen.txt ./")
    exit(1)

# 已知的file://包对应的PyPI版本映射
file_to_pypi_mapping = {
    'psutil': '6.1.1',          # 从你的汇总中看到的版本
    'PyYAML': '6.0.2',          # 常见版本
    'tqdm': '4.67.2',           # 常见版本
    'requests': '2.32.3',       # 常见版本
    'certifi': '2025.7.22',     # 最新版本
    'charset-normalizer': '3.4.1',
    'click': '8.1.8',
    'idna': '3.10',
    'urllib3': '2.3.0',
    'attrs': '25.1.0',
    'jsonschema': '4.25.1',
    'packaging': '25.0',
    'platformdirs': '4.5.0',
    'typing_extensions': '4.13.0',
    'MarkupSafe': '3.0.3',
    'Jinja2': '3.1.5',
    'cryptography': '45.0.0',
    'cffi': '1.18.1',
    'pycparser': '2.24',
    'Pygments': '2.19.1',
    'pytz': '2025.1',
    'beautifulsoup4': '4.12.3',
    'soupsieve': '2.6',
    'filelock': '3.17.0',
    'sympy': '1.14.0',
    'mpmath': '1.3.0',
}

# 分类依赖
regular_deps = []
git_deps = []
file_deps_converted = []
skipped_deps = []

for line in lines:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    
    if '@ file://' in line:
        # 提取包名
        pkg_match = re.match(r'^([a-zA-Z0-9_.-]+)\s*@', line)
        if pkg_match:
            pkg_name = pkg_match.group(1)
            # 检查是否在映射表中
            if pkg_name in file_to_pypi_mapping:
                file_deps_converted.append(f"{pkg_name}=={file_to_pypi_mapping[pkg_name]}")
            else:
                # 对于未知的包，添加注释
                skipped_deps.append(f"# {line} - 需要确定PyPI版本")
    elif '@ git+' in line:
        git_deps.append(line)
    else:
        regular_deps.append(line)

# 输出统计
print(f"处理完成：")
print(f"- 常规依赖: {len(regular_deps)} 个")
print(f"- Git依赖: {len(git_deps)} 个")
print(f"- 转换的file://依赖: {len(file_deps_converted)} 个")
print(f"- 跳过的依赖: {len(skipped_deps)} 个")

# 创建清理后的requirements文件
with open('requirements-cleaned.txt', 'w') as f:
    f.write("# 核心ML框架\n")
    for dep in sorted([d for d in regular_deps if any(pkg in d for pkg in ['torch', 'tensorflow', 'scikit'])]):
        f.write(dep + '\n')
    
    f.write("\n# 计算机视觉和图像处理\n")
    for dep in sorted([d for d in regular_deps if any(pkg in d for pkg in ['ultralytics', 'opencv', 'pillow', 'matplotlib', 'seaborn'])]):
        f.write(dep + '\n')
    
    f.write("\n# 数据处理\n")
    for dep in sorted([d for d in regular_deps if any(pkg in d for pkg in ['numpy', 'pandas', 'scipy'])]):
        f.write(dep + '\n')
    
    f.write("\n# Web框架和服务\n")
    for dep in sorted([d for d in regular_deps if any(pkg in d for pkg in ['flask', 'gunicorn', 'werkzeug'])]):
        f.write(dep + '\n')
    
    f.write("\n# 从file://转换的依赖\n")
    for dep in sorted(file_deps_converted):
        f.write(dep + '\n')
    
    f.write("\n# Git依赖（Label Studio相关）\n")
    for dep in git_deps:
        f.write(dep + '\n')
    
    f.write("\n# 其他依赖\n")
    # 添加所有未分类的依赖
    categorized = set()
    for category in [['torch', 'tensorflow', 'scikit'], ['ultralytics', 'opencv', 'pillow', 'matplotlib', 'seaborn'], 
                     ['numpy', 'pandas', 'scipy'], ['flask', 'gunicorn', 'werkzeug']]:
        for dep in regular_deps:
            if any(pkg in dep for pkg in category):
                categorized.add(dep)
    
    for dep in sorted(regular_deps):
        if dep not in categorized:
            f.write(dep + '\n')

print(f"\n清理后的文件已保存到: requirements-cleaned.txt")

# 显示关键包版本
print("\n关键包版本：")
key_packages = {
    'torch': None,
    'ultralytics': None,
    'numpy': None,
    'opencv-python': None,
    'gunicorn': None,
    'Flask': None,
}

all_deps = regular_deps + file_deps_converted
for dep in all_deps:
    for pkg in key_packages:
        if dep.startswith(pkg + '=='):
            key_packages[pkg] = dep
            
for pkg, version in key_packages.items():
    if version:
        print(f"  {version}")
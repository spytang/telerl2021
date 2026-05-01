# VS Code LaTeX 编译配置使用指南

## 📋 配置说明

已为你创建了完整的 VS Code 工作区配置，包括：

### 文件结构
```
.vscode/
├── settings.json      ← LaTeX Workshop 扩展设置
├── tasks.json         ← 编译任务定义
└── extensions.json    ← 推荐安装的扩展
```

---

## 🚀 快速开始

### 方式1️⃣：使用 LaTeX Workshop 扩展（推荐）

#### 1. 安装 LaTeX Workshop 扩展
- 打开 VS Code
- 点击左侧 **扩展** 图标（或 `Cmd+Shift+X`）
- 搜索 `latex-workshop`
- 安装 **LaTeX Workshop** by James Yu
- 或直接点击工作区中的推荐横幅安装

#### 2. 编译论文
打开 `main.tex` 文件后，有以下选择：

**选项A：使用命令面板编译**
- 按 `Cmd+Shift+P` 打开命令面板
- 输入 `LaTeX: Build` 或 `编译`
- 选择配置的编译方案：
  - ✅ **完整编译 (XeLaTeX + Biber)** - 包含参考文献
  - 🚀 **快速编译 (XeLaTeX)** - 仅编译，快速预览

**选项B：使用快捷键编译**
- 按 `Cmd+Alt+B` 进行默认编译（完整编译）

**选项C：侧边栏编译**
- 左侧会显示 **TEX** 标签
- 点击 **Build LaTeX project** 按钮

#### 3. 查看 PDF
- 编译完成后自动打开 PDF
- 或按 `Cmd+Alt+V` 查看 PDF

---

### 方式2️⃣：使用任务系统

#### 1. 打开任务菜单
- 按 `Cmd+Shift+P` 打开命令面板
- 输入 `Tasks: Run Task`

#### 2. 选择任务
```
XeLaTeX 完整编译 (含参考文献)  ← 首选推荐
XeLaTeX 快速编译               ← 仅检查语法
Biber 处理参考文献              ← 单独处理参考文献
清理编译产物                   ← 清理临时文件
```

#### 3. 查看输出
编译结果会在底部终端显示

---

### 方式3️⃣：直接在终端编译

```bash
# 完整编译（推荐）
xelatex -interaction=nonstopmode -file-line-error main.tex && \
biber main && \
xelatex -interaction=nonstopmode -file-line-error main.tex && \
xelatex -interaction=nonstopmode -file-line-error main.tex

# 快速编译
xelatex main.tex
```

---

## 📝 配置详解

### settings.json 说明

```json
{
  "latex-workshop.latex.recipes": [    // 编译流程配置
    {
      "name": "完整编译 (XeLaTeX + Biber)",
      "tools": ["xelatex", "biber", "xelatex", "xelatex"]
      // 依次运行：xelatex → biber → xelatex → xelatex
    }
  ],
  "latex-workshop.latex.tools": [      // 编译工具配置
    {
      "name": "xelatex",
      "command": "xelatex",            // 编译引擎
      "args": ["-interaction=nonstopmode", ...]  // 编译参数
    }
  ],
  "latex-workshop.view.pdf.viewer": "tab",  // PDF在标签页显示
  "latex-workshop.latex.autoBuild.run": "onSave"  // 自动保存时编译
}
```

---

## ⚙️ 常见问题

### Q1: 编译很慢？
- 第一次编译会比较慢（处理参考文献）
- 后续只改动正文可以用 **快速编译 (XeLaTeX)** 

### Q2: 怎样才能包含参考文献？
- 必须使用 **完整编译** 方案
- 流程是：xelatex → biber → xelatex → xelatex
- 单独编译一次 xelatex 不会显示参考文献

### Q3: PDF 没有自动更新？
- 打开 `main.pdf` 文件
- 确保 **autoBuild.run** 设置为 `onSave`
- 编译后自动刷新

### Q4: 编译出错怎么办？
- 查看 **问题** 面板（`Cmd+Shift+M`）
- 或查看底部的编译输出
- 错误会标记在代码编辑器中

### Q5: 需要安装什么？
- **XeTeX**：用于支持中文
- **Biber**：用于处理参考文献
- **LaTeX Workshop**：VS Code 扩展

---

## 🎯 推荐工作流程

1. **编辑论文**
   - 在 `main.tex` 或 `converted_clean_body.tex` 中编辑

2. **保存文件**
   - `Cmd+S` 保存
   - 若启用 autoBuild，自动编译

3. **查看效果**
   - `Cmd+Alt+V` 或点击 PDF 标签页
   - 查看编译后的效果

4. **修改参考文献**
   - 编辑 `references.bib`
   - 需要运行 **完整编译** 才能生效

5. **最终提交前**
   - 运行一次 **完整编译**
   - 确保所有参考文献、交叉引用正确

---

## 🔧 进阶配置（可选）

### 启用自动编译
编辑 `settings.json`，添加：
```json
"latex-workshop.latex.autoBuild.run": "onFileChange"
```
这样每次保存都会自动完整编译（可能比较慢）

### 自定义编译参数
编辑 `tasks.json` 中的 `args` 数组，如添加：
- `-file-line-error`：显示错误的文件名和行号
- `-interaction=nonstopmode`：遇到错误不暂停

---

## 💡 快速参考

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 打开命令面板 | `Cmd+Shift+P` | 输入命令执行操作 |
| 编译 | `Cmd+Alt+B` | 使用默认编译方案 |
| 查看 PDF | `Cmd+Alt+V` | 在横排查看器中预览 |
| 显示问题 | `Cmd+Shift+M` | 查看编译错误和警告 |
| 保存文件 | `Cmd+S` | 保存并触发自动编译 |
| 打开终端 | `Ctrl+`` | 手动运行编译命令 |

---

## ✅ 下一步

1. 在 VS Code 中打开此工作区文件夹
2. 安装推荐的 **LaTeX Workshop** 扩展
3. 打开 `main.tex`
4. 按 `Cmd+Alt+B` 编译
5. 按 `Cmd+Alt+V` 查看 PDF！

祝你编译顺利！🎉

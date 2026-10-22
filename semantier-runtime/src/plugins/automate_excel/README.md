# Automate Excel

一键自动化处理 Excel：读取、写入、合并、转换、校验与格式设置（.xlsx/.xls 与 CSV）。

**开源协议**：[MIT](LICENSE) · 详见 [开源说明.md](开源说明.md)

## 功能概览

- **读写**：openpyxl / pandas 读写 .xlsx，保留格式或只处理数据
- **合并与拆分**：多文件/多 sheet 合并；按行数或按列取值拆分
- **转换**：Excel ↔ CSV、按条件筛选、去重、聚合、选择/重命名列、两表/多表 VLOOKUP、转置、模板填充（{{列名}}）
- **格式**：重命名 sheet、条件格式（大于/小于/重复值/色阶）、列设为文本（避免科学计数法）
- **校验**：必须列、重复键、空行
- **批量**：对目录下多个文件统一处理

## 使用方式

### 在 Cursor / OpenClaw 中

对话里提到 Excel、表格、.xlsx、合并、筛选、科学计数法等时，Agent 会加载本 skill，根据你的需求选脚本或写代码。**不知道用哪个脚本时**：打开 [SKILL.md](SKILL.md)，看「你想做的事 → 调用的脚本」表。

### 本地直接运行脚本

先安装依赖，再执行对应脚本（每个脚本支持 `--help`）：

```bash
pip install -r scripts/requirements.txt
```

| 你想做的事           | 命令示例 |
|----------------------|----------|
| 多文件/多 sheet 合并 | `python scripts/merge_sheets.py --inputs a.xlsx b.xlsx --output out.xlsx` |
| 导出 CSV             | `python scripts/excel_to_csv.py --input file.xlsx --output file.csv` |
| CSV 转 Excel         | `python scripts/csv_to_excel.py --input a.csv --output a.xlsx` |
| 按条件筛选           | `python scripts/filter_excel.py --input data.xlsx --where "状态=已完成" --output out.xlsx` |
| 按行数/按列拆分      | `python scripts/split_excel.py --input large.xlsx --by-rows 5000 --output-dir ./chunks` |
| 去重                 | `python scripts/deduplicate_excel.py --input data.xlsx --keys 编号 --output unique.xlsx` |
| 分组聚合             | `python scripts/aggregate_excel.py --input sales.xlsx --group-by 地区 --agg "销售额:sum" --output summary.xlsx` |
| 校验列与重复         | `python scripts/validate_excel.py --input data.xlsx --require-cols 名称,金额 --key-cols 编号` |
| 选择/重命名列        | `python scripts/select_columns.py --input data.xlsx --columns 姓名,金额 --output out.xlsx` |
| 两表按键合并         | `python scripts/merge_tables.py --left a.xlsx --right b.xlsx --on 编号 --output merged.xlsx` |
| 多表 VLOOKUP         | `python scripts/vlookup_multi.py --main 订单.xlsx --lookups "客户.xlsx:客户ID" "产品.xlsx:产品ID" --output out.xlsx` |
| 行列转置             | `python scripts/transpose_excel.py --input data.xlsx --output transposed.xlsx` |
| 模板填充 {{列名}}    | `python scripts/template_fill.py --template t.xlsx --data d.csv --output filled.xlsx` |
| 重命名 sheet         | `python scripts/rename_sheets.py --input file.xlsx --rename "Sheet1:汇总" --output out.xlsx` |
| 条件格式             | `python scripts/format_conditional.py --input data.xlsx --column C --rule gt --value 100 --fill red --output out.xlsx` |
| 列设为文本（防科学计数法） | `python scripts/format_columns_as_text.py --input data.xlsx --columns 身份证号,订单号 --output out.xlsx` |

完整参数见各脚本 `--help` 或 [reference.md](reference.md)。

## 脚本一览（16 个）

| 脚本 | 功能 |
|------|------|
| merge_sheets.py | 多 Excel / 多 sheet 合并为一张表 |
| excel_to_csv.py | 指定 sheet 导出为 CSV |
| csv_to_excel.py | CSV 转 Excel（单/多 CSV → 多 sheet） |
| filter_excel.py | 按条件筛选（=、>、<、~ 包含） |
| split_excel.py | 按行数分片或按某列取值拆分 |
| deduplicate_excel.py | 按列去重 |
| aggregate_excel.py | 按列分组聚合（sum/count/mean/min/max） |
| validate_excel.py | 校验必须列、重复键、空行 |
| select_columns.py | 选择/重命名/排序列 |
| merge_tables.py | 两表按键合并（left/inner/outer） |
| vlookup_multi.py | 多表 VLOOKUP（主表 + 多个查找表） |
| transpose_excel.py | 行列转置 |
| template_fill.py | 模板 {{列名}} 按行填充 |
| rename_sheets.py | 重命名工作表（原名:新名、前缀/后缀） |
| format_conditional.py | 条件格式（gt/lt/介于/重复值/色阶） |
| format_columns_as_text.py | 列设为文本，避免科学计数法 |

## 依赖

- Python 3.8+
- openpyxl、pandas、xlrd（仅读 .xls 时需要）

```bash
pip install -r scripts/requirements.txt
```

## 故障排除

- **找不到模块**：在项目根或 skill 目录执行 `pip install -r scripts/requirements.txt`
- **编码**：导出 CSV 默认 `utf-8-sig`，Excel 可正确识别中文
- **大文件**：可用 `openpyxl.load_workbook(..., read_only=True)` 或 pandas 分块

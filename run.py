"""项目统一入口；训练、选参和最终评估编排代码位于训练流程目录。"""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "02_训练流程/04_训练模型/train.py"), run_name="__main__")

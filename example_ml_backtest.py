#!/usr/bin/env python3
"""
示例：在回测中使用ML波动率预测

使用步骤：
1. 先训练模型: python train_volatility_model.py --symbol AAPL --use-synthetic
2. 运行回测: python example_ml_backtest.py
"""

import sys
sys.path.insert(0, '/mnt/harddisk/lwb/options-trading-platform')

from core.backtesting.engine import BacktestEngine
from core.ml.inference.predictor import VolatilityPredictor


def run_backtest_with_ml(symbol="AAPL", use_ml=True):
    """运行回测，可选择是否使用ML预测"""
    
    # 初始化波动率预测器
    vol_predictor = None
    if use_ml:
        vol_predictor = VolatilityPredictor()
        if vol_predictor.is_ready():
            print(f"✅ ML波动率预测器已加载")
        else:
            print(f"⚠️ ML模型未找到，将使用历史波动率作为IV代理")
            print(f"   请先运行: python train_volatility_model.py --symbol {symbol} --use-synthetic")
    
    # 创建回测引擎
    engine = BacktestEngine(data_client=None, vol_predictor=vol_predictor)
    
    # 回测参数
    params = {
        "strategy": "sell_put",
        "symbol": symbol,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_capital": 100000,
        "dte_min": 30,
        "dte_max": 45,
        "delta_target": 0.30,
        "profit_target_pct": 50,
        "stop_loss_pct": 200,
        "use_synthetic_data": True,  # 使用合成数据进行测试
    }
    
    print(f"\n{'='*60}")
    print(f"运行回测: {symbol}")
    print(f"使用ML预测: {use_ml and vol_predictor and vol_predictor.is_ready()}")
    print(f"{'='*60}\n")
    
    # 执行回测
    try:
        results = engine.run(params)
        
        # 打印结果
        metrics = results["metrics"]
        print(f"\n回测结果:")
        print(f"  总收益率: {metrics.get('total_return', 0):.2f}%")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"  交易次数: {len(results.get('trades', []))}")
        
        return results
        
    except Exception as e:
        print(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_ml_vs_hv():
    """对比ML预测 vs 历史波动率的效果"""
    
    print("\n" + "="*60)
    print("对比测试: ML预测 vs 历史波动率")
    print("="*60)
    
    # 1. 使用历史波动率
    results_hv = run_backtest_with_ml(use_ml=False)
    
    # 2. 使用ML预测（如果模型存在）
    results_ml = run_backtest_with_ml(use_ml=True)
    
    # 对比结果
    if results_hv and results_ml:
        print("\n" + "="*60)
        print("对比结果")
        print("="*60)
        
        hv_return = results_hv["metrics"].get("total_return", 0)
        ml_return = results_ml["metrics"].get("total_return", 0)
        
        print(f"历史波动率 (HV) 收益率: {hv_return:.2f}%")
        print(f"ML预测收益率: {ml_return:.2f}%")
        print(f"差异: {ml_return - hv_return:+.2f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ML回测示例")
    parser.add_argument("--symbol", default="AAPL", help="股票代码")
    parser.add_argument("--compare", action="store_true", help="对比ML vs HV")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_ml_vs_hv()
    else:
        run_backtest_with_ml(symbol=args.symbol, use_ml=True)

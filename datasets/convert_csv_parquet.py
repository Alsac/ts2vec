import pandas as pd
import argparse

def csv2parquet(file_path, save_path):
    """
    将csv文件格式转换为parquet
    file_path:输入csv文件路径，比如D:\home\GitHub\Astock_daily.csv'，其中Astock_daily.csv为文件名
    save_path:输入要保持的parquet格式文件路径，比如D:\home\GitHub\Astock_daily.parquet'，其中Astock_daily.parquet为文件名
    """
    data = pd.read_csv(file_path, index_col=0)
    data.to_parquet(save_path, compression='gzip')
    return None 

def parquet2csv(file_path, save_path):
    """
    将parquet文件格式转换为csv
    file_path:parquet，比如D:\home\GitHub\Astock_daily.parquet'
    save_path:输入要保持的parquet格式文件路径
    """
    data = pd.read_parquet(file_path)
    data.to_csv(save_path)
    return None
if __name__ == '__main__':
    try:
        import pyarrow
    except:
        import os
        print('缺少pyarrow模块，准备下载...')
        os.system('pip install pyarrow')
        print('下载完成！')
    parser = argparse.ArgumentParser(description='csv与parquet互相转换的脚本')
    parser.add_argument("--file_path", type=str, default=None)  # r'D:\home\GitHub\ts2vec\datasets\BTCUSDT.csv'
    parser.add_argument("--save_path", type=str, default=None)  # r'D:\home\GitHub\ts2vec\datasets\BTCUSDT.parquet'
    args = parser.parse_args()
    file_format = args.file_path.split('.')[-1]
    if file_format == 'csv':
        csv2parquet(args.file_path, args.save_path)
    elif file_format == 'parquet':
        parquet2csv(args.file_path, args.save_path)
import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets

def main():
    # 参数设置
    parser = argparse.ArgumentParser(description='Image Retrieval Demo')
    parser.add_argument('--test_dir', default='./data/masked_test', type=str, help='Test data directory')
    parser.add_argument('--query_list', default='./challenge/query_street_name.txt', type=str, help='Query image list file')
    parser.add_argument('--output_file', default='./challenge/answer.txt', type=str, help='Output result file')
    args = parser.parse_args()

    # 设置数据集路径
    gallery_name = 'gallery_satellite'
    query_name = 'query_street'
    image_datasets = {
        gallery_name: datasets.ImageFolder(os.path.join(args.test_dir, gallery_name)),
        query_name: datasets.ImageFolder(os.path.join(args.test_dir, query_name))
    }

    # 加载特征和标签
    result = scipy.io.loadmat('pytorch_result.mat')
    query_feature = torch.FloatTensor(result['query_f']).cuda()
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
    gallery_label = result['gallery_label'][0]

    # 创建图像路径到索引的映射
    query_img_paths = [os.path.basename(p) for p, _ in image_datasets[query_name].imgs]
    gallery_img_paths = [os.path.basename(p) for p, _ in image_datasets[gallery_name].imgs]

    # 读取查询列表
    with open(args.query_list, 'r') as f:
        query_names = [line.strip() for line in f.readlines() if line.strip()]

    # 排序函数
    def sort_img(qf, ql, gf, gl):
        query = qf.view(-1, 1)
        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        index = np.argsort(score)[::-1]  # 从大到小排序
        junk_index = np.argwhere(gl == -1)
        mask = np.in1d(index, junk_index, invert=True)
        return index[mask]

    # 处理每个查询图像
    results = []
    for query_name in query_names:
        try:
            # 查找查询图像的索引
            query_idx = query_img_paths.index(query_name)
            # 获取排序结果
            sorted_idx = sort_img(query_feature[query_idx], query_label[query_idx], 
                            gallery_feature, gallery_label)
            # 获取top10图像名称（不含扩展名）
            top10_names = [os.path.splitext(gallery_img_paths[i])[0] for i in sorted_idx[:10]]
            # 添加到结果列表（仅10个匹配结果）
            results.append('\t'.join(top10_names))
        except ValueError:
            print(f"Warning: Query image {query_name} not found in dataset")
            results.append('\t'.join(['']*10))

    # 保存结果到文件
    with open(args.output_file, 'w') as f:
        f.write('\n'.join(results))
    print(f"Results saved to {args.output_file}")

if __name__ == '__main__':
    main()
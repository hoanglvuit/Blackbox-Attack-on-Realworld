import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Hiển thị ảnh từ file .npy")
    parser.add_argument("--npy_path", default="log/ex.npy", type=str, help='Đường dẫn tới file .npy')
    args = parser.parse_args()

    # Kiểm tra file tồn tại
    if not os.path.isfile(args.npy_path):
        raise FileNotFoundError(f"Không tìm thấy file: {args.npy_path}")

    # Load dữ liệu từ file
    data = np.load(args.npy_path, allow_pickle=True).item()
    adv = data['adversary']
    orig = data['orig']

    # Hiển thị ảnh
    plt.imshow(adv)
    plt.axis('off')
    plt.title("Ảnh RGB (0–1)")
    plt.show()

if __name__ == '__main__':
    main()

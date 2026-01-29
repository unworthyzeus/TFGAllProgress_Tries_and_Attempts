import os

path = r"Paris\Paris_3_image_128_AoA\Paris_3_128_BS10_0_0_AoA.png"
print(f"Checking: {path}")
print(f"Exists: {os.path.exists(path)}")

abs_path = os.path.abspath(path)
print(f"Abs path: {abs_path}")
print(f"Exists abs: {os.path.exists(abs_path)}")

# List dir
dir_path = r"Paris\Paris_3_image_128_AoA"
if os.path.exists(dir_path):
    print(f"Dir exists. Listing first 5:")
    print(os.listdir(dir_path)[:5])
else:
    print("Dir does not exist")

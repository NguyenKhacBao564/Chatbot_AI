import os

def dump_entire_project_to_single_txt(source_folder, output_file):
    # Danh sách các thư mục cần loại bỏ hoàn toàn
    excluded_dirs = ['node_modules', '.vscode', 'package-lock.json','.next','.git', '.github', 'venv']
    # Danh sách các file cần bỏ qua
    excluded_files = ['venv', '.gitignore', 'Dockerfile', 'docker-compose.yaml', 'LICENSE', '.next','package-lock.json','settings.json']

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for root, dirs, files in os.walk(source_folder):
            # Loại bỏ các thư mục không muốn duyệt
            dirs[:] = [d for d in dirs if d not in excluded_dirs]

            for file in files:
                # Nếu file nằm trong danh sách bỏ qua thì skip
                if file in excluded_files:
                    continue

                # Nếu muốn bỏ qua một số file kiểu binary (ví dụ .png, .jpg, .exe, .dll)
                if file.endswith(('.png', '.jpg', '.exe', '.dll', '.sample')):
                    continue

                file_path = os.path.join(root, file)

                # Đọc nội dung file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Lỗi khi đọc file {file_path}: {e}")
                    continue

                # Ghi tiêu đề (đường dẫn file) trước, rồi ghi nội dung
                out_f.write(f"=== FILE: {file_path} ===\n")
                out_f.write(content)
                out_f.write("\n\n")  # Dòng trống phân tách

                print(f"Đã ghi nội dung file: {file_path}")

# Chỉ định đường dẫn thư mục gốc và file text đầu ra
# source_folder = "/Users/nguyen_bao/Documents/Project/ttcs/tourGuide123"
# source_folder = "/Users/nguyen_bao/Documents/PTIT/Junior_2/Thực tập cơ sở/TourBooking__React"
# source_folder = "/Users/nguyen_bao/Documents/PTIT/Junior_2/ltw/tour-booking-web"
source_folder = "D:\PROJECT\Machine Learning\Chat Bot\dataset\Chatbot_AI2\Chatbot_AI"
# source_folder = "/Users/nguyen_bao/Documents/PTIT/AI_repo/gen-ai-travel-advisor"
output_file   = "D:\PROJECT\Machine Learning\Chat Bot\codeAI_repo.txt"

dump_entire_project_to_single_txt(source_folder, output_file)

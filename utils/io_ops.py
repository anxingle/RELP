import re

import os
import shutil
import zipfile
import time
import subprocess

def radar_download(dir, radar):
    download_completion = 0

    class ProgressPrintingEnclosureTransferDelegate(radarclient.RadarEnclosureTransferDelegate):
        def update_transfer_progress(self, enclosure, bytes_transferred, bytes_total):
            print('Enclosure {} {}%'.format(enclosure, int((bytes_transferred * 100) / bytes_total)))

    new_folder = 'download'
    dir = os.path.join(dir, new_folder)
    system_identifier = radarclient.ClientSystemIdentifier('RadarClient', '1.0')
    authentication_strategy = radarclient.AuthenticationStrategySPNego()
    client = radarclient.RadarClient(authentication_strategy, client_system_identifier=system_identifier)
    radar_id = radar
    radar = client.radar_for_id(radar_id)
    attachments_to_download = []
    attachments = radar.attachments.items()
    pics = radar.pictures.items()
    system_time = datetime.now()

    for i in attachments:
        ori_upload_time = i.addedAt
        IR_upload_time = ori_upload_time + timedelta(hours=8)
        IR_upload_time = IR_upload_time.replace(tzinfo=None)
        head, last_path_component = os.path.split(i.fileName)
        file_root, file_extension = os.path.splitext(last_path_component)
        print("File extension:", file_extension)
        # print("{} {}".format(head, last_path_component))
        time_difference = system_time - IR_upload_time
        if time_difference < timedelta(hours=1448) and '.zip' in file_extension:
            attachments_to_download.append((radar, last_path_component, i))
            print("File " + last_path_component + " will be a download candidate")
        else:
            print("The difference is 24 hours or more or not a zip file.")
    # print(attachments_to_download)
    download_directory = dir
    if not os.path.isdir(download_directory):
        os.makedirs(download_directory)

    for radar, last_path_component, attachment in attachments_to_download:
        download_path = os.path.join(download_directory, last_path_component)
        if os.path.exists(download_path) and os.stat(download_path).st_size == attachment.fileSize:
            print('Not downloading attachment {} from {}, file exists'.format(attachment.fileName, radar.id))
            continue
        print('Downloading attachment {} from {} to {}'.format(attachment.fileName, radar.id, download_path))
        with open(download_path, 'wb') as f:
            attachment.transfer_delegate = ProgressPrintingEnclosureTransferDelegate()
            f.write(attachment.content())
    download_completion = 1
    if download_completion == 1:
        unzip(dir)
    return dir


def radar_upload(path, radar_id):
    class ProgressPrintingEnclosureTransferDelegate(radarclient.RadarEnclosureTransferDelegate):
        def update_transfer_progress(self, enclosure, bytes_transferred, bytes_total):
            print('Enclosure {} {}%'.format(enclosure, int((bytes_transferred * 100) / bytes_total)))

    system_identifier = radarclient.ClientSystemIdentifier('RadarClient', '1.0')
    authentication_strategy = radarclient.AuthenticationStrategySPNego()
    client = radarclient.RadarClient(authentication_strategy, client_system_identifier=system_identifier)
    radar = client.radar_for_id(radar_id)
    attachment = radar.new_attachment(os.path.basename(path))
    attachment.transfer_delegate = ProgressPrintingEnclosureTransferDelegate()
    attachment.set_upload_file(open(path, mode='rb'))
    radar.attachments.add(attachment)
    radar.commit_changes()


def unzip(folder_path):
    zip_file_count = 0
    destin_file = 'archives'
    desfile_parent = os.path.dirname(folder_path)
    destination_folder_path = os.path.join(desfile_parent, destin_file)
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是ZIP文件
        if filename.endswith('.zip'):
            file_root, file_extension = os.path.splitext(filename)
            zip_file_count += 1
            # 构造完整的文件路径
            zip_file_path = os.path.join(folder_path, filename)
            print(f"Found ZIP file: {zip_file_path}")

            # 解压ZIP文件
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # 指定解压到的目标目录
                extract_to_path = os.path.join(folder_path, file_root)
                # 创建目标目录如果它不存在
                if not os.path.exists(extract_to_path):
                    os.makedirs(extract_to_path)
                # 解压所有文件到目标目录
                zip_ref.extractall(extract_to_path)
                print(f"Extracted {filename} to {extract_to_path}")

            move_file(zip_file_path, destination_folder_path)

    remove_MACOSX(folder_path)
    # 打印ZIP文件总数
    print(f"Total ZIP files found: {zip_file_count}")


def zip_folder(folder_path, file_name):
    # 创建一个ZipFile对象，并设置模式为写
    output_path = os.path.join(folder_path, file_name)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # 创建文件的完整路径
                file_path = os.path.join(root, file)
                # 创建文件在ZIP中的路径
                in_zip_path = os.path.relpath(file_path, os.path.join(folder_path, '..'))
                # 将文件添加到ZIP中
                zipf.write(file_path, in_zip_path)
    return file_path


def remove_MACOSX(directory):
    # 遍历目录中的所有文件和文件夹
    for root, dirs, files in os.walk(directory, topdown=False):
        # 检查是否存在 __MACOSX 文件夹
        if '__MACOSX' in dirs:
            # 构造 __MACOSX 文件夹的完整路径
            macosx_folder = os.path.join(root, '__MACOSX')
            # 删除 __MACOSX 文件夹
            shutil.rmtree(macosx_folder)
            print(f"Removed __MACOSX folder from {root}")


def move_file(file, destination_folder_path):
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    elif os.path.exists(destination_folder_path):
        shutil.rmtree(destination_folder_path)
        os.makedirs(destination_folder_path)
    shutil.move(file, destination_folder_path)


def move_file_keepfolder(file, destination_folder_path):
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)
    shutil.move(file, destination_folder_path)


def delete_empty_folders(folder_path):
    # iterate a few times
    for i in range(2):
        for root, dirs, files in os.walk(folder_path):
            # check the size of each folder, delete those smaller than threshold
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if os.path.getsize(dir_path) < 1000:
                    shutil.rmtree(dir_path)


def path_creator(main_path, subpath_list):
    abs_path = main_path
    for path in subpath_list:
        abs_path = os.path.join(abs_path, str(path))
    return abs_path


def Count_Subfolders(root_dir, required_duplicates):
    # 定义允许的文件扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    excel_extensions = {'.xlsx', '.xls', '.csv'}
    all_extensions = image_extensions | excel_extensions

    result_folders = []

    # 遍历根目录下的所有文件夹
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 跳过隐藏文件夹
        if os.path.basename(dirpath).startswith('.'):
            continue

        # 获取当前文件夹中的所有合法文件
        valid_files = [
            os.path.splitext(f)
            for f in filenames
            if not f.startswith('.') and
               os.path.splitext(f)[1].lower() in all_extensions
        ]

        if not valid_files:
            continue

        # 只有当required_duplicates > 1时才检查文件类型共存
        if required_duplicates > 1:
            has_image = any(ext.lower() in image_extensions for _, ext in valid_files)
            has_excel = any(ext.lower() in excel_extensions for _, ext in valid_files)

            if not (has_image and has_excel):
                continue

        # 使用字典统计同名文件
        filename_counts = {}
        for base_name, _ in valid_files:
            filename_counts[base_name] = filename_counts.get(base_name, 0) + 1

        # 检查是否有文件名重复次数等于required_duplicates的文件
        if any(count >= required_duplicates for count in filename_counts.values()):
            result_folders.append(dirpath)

    return result_folders


def create_folder(Tier1_path, Tier2_path):
    if Tier2_path is None:
        Tier2_path = ""
    new_path = os.path.join(Tier1_path, Tier2_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    return new_path


def two_steps_path_join(main_path, middle_str, sub_path):
    step1_path = os.path.join(main_path, middle_str)
    step2_path = os.path.join(step1_path, sub_path)
    return step2_path


def extract_zip_without_macosx(zip_path, extract_to):
    #Handles __MACOSX and .DS_Store.
    #Flattens a redundant top-level folder (e.g., bumper/bumper/… → bumper/…).
    #Safe with nested subdirectories.

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Filter out macOS metadata
        file_list = [
            f for f in zip_ref.namelist()
            if not f.startswith('__MACOSX') and not f.endswith('.DS_Store')
        ]

        # Identify top-level directories (only for files)
        top_dirs = set()
        for f in file_list:
            if not f.endswith('/'):  # skip directory entries
                parts = f.split('/')
                if parts:
                    top_dirs.add(parts[0])

        # Determine if we should flatten a single top-level folder
        common_prefix = ''
        if len(top_dirs) == 1:
            common_prefix = top_dirs.pop() + '/'

        # Extract each file
        for member in file_list:
            if member.endswith('/'):  # skip directory entries
                continue

            if common_prefix and member.startswith(common_prefix):
                rel_path = member[len(common_prefix):]
            else:
                rel_path = member

            target_path = os.path.join(extract_to, rel_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            with zip_ref.open(member) as source, open(target_path, 'wb') as target:
                target.write(source.read())



# --- Path Parsers ---
def extract_config_group_from_path(image_path):
    """
    从图片路径中提取 Config_Group 信息。
    例如: /Users/RELP_3/Pics/macbook/macbook air/B3/1.jpg -> 返回 'B3'
          /Users/RELP_3/Pics/macbook/macbook air/Remaining/1.jpg -> 返回 'Remaining'

    Args:
        image_path: 图片的完整路径

    Returns:
        str: Config_Group 值，如果没有找到则返回空字符串
    """
    if not image_path:
        return ""

    # 将路径按分隔符分割
    path_parts = image_path.replace('\\', '/').split('/')

    # 遍历路径的每个部分，寻找可能的 Config_Group 值
    for part in path_parts:
        part = part.strip()
        # 匹配模式1: 1-2个大写字母后跟1-2个数字（如 B3, AB12, A1）
        if re.match(r'^[A-Z]{1,2}\d{1,2}$', part):
            return part
        # 匹配模式2: 常用 Config_Group 值（如 Remaining, All, General 等）
        if part.lower() in ['remaining', 'all', 'general', 'other']:
            return part

    return ""


def extract_product_side_from_path(image_path):
    """
    从图片路径中提取 Product_Side 信息。
    根据路径中的关键字匹配 Product_Side，支持多种命名格式：
    - Side1, Side2, Side3...
    - Side_1, Side_2...
    - R360, R361...
    - 其他自定义标识

    匹配规则：
    1. 如果路径中包含 Project_Code（如 R360）和 Product_Side（如 Side1），则组合返回
    2. 如果只包含 Product_Side，则直接返回
    3. 如果没有匹配到，返回空字符串

    例如:
    - /path/R360/Side1/1.jpg -> 返回 'R360_Side1'
    - /path/R360/Side1/ -> 返回 'R360_Side1'
    - /path/Side1/1.jpg -> 返回 'Side1'
    - /path/to/image.jpg -> 返回 ''

    Args:
        image_path: 图片的完整路径

    Returns:
        str: Product_Side 值（可能包含 Project_Code 前缀），如果没有找到则返回空字符串
    """
    if not image_path:
        return ""

    # 将路径按分隔符分割
    path_parts = image_path.replace('\\', '/').split('/')

    project_code = None
    product_side = None

    # 遍历路径的每个部分
    for part in path_parts:
        part = part.strip()
        if not part:
            continue

        part_lower = part.lower()

        # 匹配 Project_Code: R 开头后跟数字（如 R360, R210）
        if re.match(r'^[Rr]\d+$', part):
            project_code = part

        # 匹配 Product_Side: Side 开头后跟数字（支持 Side1, Side_1, side-1 等格式）
        side_match = re.match(r'^side[_-]?(\d+)$', part_lower)
        if side_match:
            product_side = f"Side{side_match.group(1)}"

    # 组合结果
    if project_code and product_side:
        return f"{project_code}_{product_side}"
    elif product_side:
        return product_side
    else:
        return ""



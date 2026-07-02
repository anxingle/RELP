import radarclient
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import sys
import time
import subprocess
import shutil
import threading

import zipfile
import atexit

# Get current directory
current_dir = Path.cwd()

# ==========================================
# macOS 防休眠/防锁屏功能 (Caffeinate)
# ==========================================
caffeinate_process = None

def prevent_sleep():
    """在 macOS 下启动 caffeinate 进程以防止休眠和亮屏"""
    global caffeinate_process
    if sys.platform == 'darwin':
        try:
            # -d: 阻止显示器休眠 (防止锁屏)
            # -i: 阻止系统空闲休眠
            # -s: 当连接电源时阻止系统休眠
            caffeinate_process = subprocess.Popen(['caffeinate', '-d', '-i', '-s'])
            print("System sleep and screen lock prevention activated (macOS).")
        except Exception as e:
            print(f"Failed to activate sleep prevention: {e}")

def allow_sleep():
    """脚本退出时终止 caffeinate 进程，恢复系统正常休眠策略"""
    global caffeinate_process
    if caffeinate_process:
        try:
            caffeinate_process.terminate()
            print("System sleep prevention deactivated. Normal sleep rules resumed.")
        except Exception:
            pass

# 注册退出清理函数：无论是正常结束还是被 Ctrl+C 中断，都会执行
atexit.register(allow_sleep)
# ==========================================

def extract_zip_without_macosx(zip_path, extract_to):
    """
    Extracts a zip file to the specified directory, ignoring __MACOSX folder.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Filter out __MACOSX
            members = [m for m in zip_ref.namelist() if not m.startswith('__MACOSX')]
            zip_ref.extractall(extract_to, members=members)
            print(f"Extracted {os.path.basename(zip_path)} to {extract_to}")
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")

def check_and_renew_kerberos_ticket():
    """
    Checks if the Kerberos ticket is valid and attempts to renew/re-login if necessary.
    This function assumes 'kinit' is available and configured for the user.
    """
    try:
        # Check if klist returns successfully
        subprocess.check_call(['klist', '-s'])
        # print("Kerberos ticket is valid.")
        
        # 即使 ticket 是 valid 的，我们也尝试续期，以确保长时间运行不断线
        # 使用丢弃输出的方式，避免污染控制台
        try:
            subprocess.check_call(['kinit', '-R'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
            
        return True
    except subprocess.CalledProcessError:
        print("Kerberos ticket expired or invalid. Attempting to renew...")
        try:
            # Attempt to renew using kinit -R (renew) first
            subprocess.check_call(['kinit', '-R'])
            print("Kerberos ticket renewed successfully.")
            return True
        except subprocess.CalledProcessError:
            print("Failed to renew ticket. Please log in manually using 'kinit' or AppleConnect.")
            # In a fully automated environment, you might use a keytab here:
            # subprocess.check_call(['kinit', '-k', '-t', '/path/to/keytab', 'principal'])
            return False

def radar_download(base_dir, radar_id, time_threshold_hours, file_formats):
    """
    Downloads files from a specified radar ID based on time threshold and file formats.
    
    Args:
        base_dir (str): Base directory to save downloaded files.
        radar_id (int): The Radar ID to download from.
        time_threshold_hours (float): Download files uploaded within this many hours.
        file_formats (list): List of file extensions to download (e.g., ['.zip', '.jpg']).
    """
    
    class ProgressPrintingEnclosureTransferDelegate(radarclient.RadarEnclosureTransferDelegate):
        def __init__(self, filename):
            self.filename = filename
            self.last_percent = -1

        def update_transfer_progress(self, enclosure, bytes_transferred, bytes_total):
            percent = int((bytes_transferred * 100) / bytes_total)
            if percent != self.last_percent:
                # Simple progress bar
                bar_length = 20
                filled_length = int(bar_length * percent // 100)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f'\r[Radar {radar_id}] Downloading {self.filename}: [{bar}] {percent}%')
                sys.stdout.flush()
                self.last_percent = percent
            if percent == 100:
                sys.stdout.write('\n')

    max_retries = 3
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            # Ensure authentication is valid before starting
            # Note: In a threaded environment, this check might overlap. 
            # Ideally, ticket renewal should be handled by a central manager or locked.
            # For simplicity, we'll assume kinit -R is safe to call concurrently or fails gracefully.
            check_and_renew_kerberos_ticket()

            system_identifier = radarclient.ClientSystemIdentifier('RadarClient', '1.0')
            authentication_strategy = radarclient.AuthenticationStrategyAppleConnect()
            client = radarclient.RadarClient(authentication_strategy, client_system_identifier=system_identifier)
            
            print(f"[Radar {radar_id}] Connecting...")
            radar = client.radar_for_id(radar_id)
            
            attachments = []
            try:
                attachments.extend(radar.attachments.items())
            except Exception as e:
                print(f"[Radar {radar_id}] Error fetching attachments: {e}")

            try:
                pictures = radar.pictures.items()
                attachments.extend(pictures)
            except Exception as e:
                print(f"[Radar {radar_id}] Error fetching pictures: {e}")
                
            system_time = datetime.now()
            
            print(f"[Radar {radar_id}] Checking {len(attachments)} items (attachments + pictures)...")
            
            attachments_to_download = []
            for i in attachments:
                ori_upload_time = i.addedAt
                # Adjust time zone if necessary (assuming +8 hours based on reference script)
                IR_upload_time = ori_upload_time + timedelta(hours=8)
                IR_upload_time = IR_upload_time.replace(tzinfo=None)
                
                head, last_path_component = os.path.split(i.fileName)
                file_root, file_extension = os.path.splitext(last_path_component)
                
                # print(f"Found file: {last_path_component} (Type: {file_extension}, Uploaded: {IR_upload_time})")

                # Check file format
                if file_extension.lower().replace('.', '') not in [fmt.lower().replace('.', '') for fmt in file_formats]:
                    continue

                time_difference = system_time - IR_upload_time
                
                # Check time threshold
                if time_difference < timedelta(hours=time_threshold_hours):
                    attachments_to_download.append((radar, last_path_component, i))
                    print(f"[Radar {radar_id}] File {last_path_component} (Type: {file_extension}) is a download candidate. Uploaded at: {IR_upload_time}")
                else:
                    # print(f"File {last_path_component} is too old ({time_difference}).")
                    pass

            if not attachments_to_download:
                print(f"[Radar {radar_id}] No files found matching criteria.")
                return

            # Create timestamped folder
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_directory = os.path.join(base_dir, timestamp_str)
            
            if not os.path.isdir(download_directory):
                os.makedirs(download_directory)
                print(f"[Radar {radar_id}] Created directory: {download_directory}")

            print(f"[Radar {radar_id}] Starting download of {len(attachments_to_download)} files to {download_directory}...")

            history_file = os.path.join(base_dir, "download_history.txt")
            downloaded_files = set()
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    downloaded_files = set(line.strip() for line in f)

            files_downloaded_in_this_session = 0

            for radar_obj, last_path_component, attachment in attachments_to_download:
                # Get the full relative path from radar (preserving folder structure)
                full_relative_path = attachment.fileName
                # Remove leading slash if present
                full_relative_path = full_relative_path.lstrip('/')
                
                # Unique identifier for the file: RadarID_FileName_FileSize (or UploadTime)
                file_identifier = f"{radar_id}_{full_relative_path}_{attachment.fileSize}"
                
                if file_identifier in downloaded_files:
                    print(f"[Radar {radar_id}] Skipping {full_relative_path}, already downloaded previously.")
                    continue

                # Create the full download path preserving folder structure
                download_path = os.path.join(download_directory, full_relative_path)
                
                # Create subdirectories if they don't exist
                download_subdir = os.path.dirname(download_path)
                if download_subdir and not os.path.exists(download_subdir):
                    os.makedirs(download_subdir)
                    print(f"[Radar {radar_id}] Created subdirectory: {download_subdir}")
                
                try:
                    with open(download_path, 'wb') as f:
                        attachment.transfer_delegate = ProgressPrintingEnclosureTransferDelegate(full_relative_path)
                        f.write(attachment.content())
                    
                    # Update history
                    # Use a lock if multiple threads might write to the same history file (unlikely if base_dir is unique per radar)
                    # But if base_dir is shared, we need locking. Assuming base_dir is unique per radar for now.
                    with open(history_file, 'a') as f:
                        f.write(f"{file_identifier}\n")
                    downloaded_files.add(file_identifier)
                    files_downloaded_in_this_session += 1
                    
                    # Auto-unzip if it's a zip file
                    if full_relative_path.lower().endswith('.zip'):
                        print(f"[Radar {radar_id}] Auto-unzipping {full_relative_path}...")
                        # Extract to the same directory as the zip file (preserving structure)
                        zip_extract_dir = os.path.dirname(download_path)
                        extract_zip_without_macosx(download_path, zip_extract_dir)
                    
                except Exception as e:
                    print(f"[Radar {radar_id}] Failed to download {attachment.fileName}: {e}")
            
            # If files were downloaded, rename folder to indicate completion
            if files_downloaded_in_this_session > 0:
                new_dir_name = download_directory + "_downloaded"
                os.rename(download_directory, new_dir_name)
                print(f"[Radar {radar_id}] Download session complete. Renamed folder to: {new_dir_name}")
                return new_dir_name
            else:
                # If empty (all skipped), maybe remove the empty folder?
                if not os.listdir(download_directory):
                    os.rmdir(download_directory)
                    print(f"[Radar {radar_id}] No new files downloaded. Removed empty directory.")
                else:
                     print(f"[Radar {radar_id}] Session complete. Files in: {download_directory}")
                return None

            # If we reach here, everything succeeded, so break the retry loop
            break

        except Exception as e:
            print(f"[Radar {radar_id}] An error occurred: {e}")
            # Check if the error is related to authentication
            if "No AppleConnect session established" in str(e) or "expired" in str(e).lower():
                print(f"[Radar {radar_id}] Authentication error detected. Attempting to re-authenticate...")
                if check_and_renew_kerberos_ticket():
                    print(f"[Radar {radar_id}] Re-authentication successful. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
                else:
                    print(f"[Radar {radar_id}] Re-authentication failed. Aborting.")
                    break
            else:
                # For other errors, we might not want to retry immediately or at all
                print(f"[Radar {radar_id}] Non-authentication error. Aborting.")
                break

# Global lock for RELP3_main.py execution
analysis_lock = threading.Lock()

def trigger_analysis(downloaded_folder, product, generation, failure_mode=None, downloading_radar=None, auto_detect_fm=False):
    """
    Triggers RELP3_main.py with the downloaded folder path and other parameters.
    
    Args:
        downloaded_folder: Path to the folder containing images
        product: Product name
        generation: Generation name
        failure_mode: Specific failure mode to process (optional if auto_detect_fm=True)
        downloading_radar: Radar ID for downloading
        auto_detect_fm: If True, don't pass --failure_mode, let RELP3_main.py detect from config
    """
    if auto_detect_fm:
        print(f"Triggering RELP3_main.py for {product} {generation} with path: {downloaded_folder} (auto-detect failure mode from config)")
    else:
        print(f"Triggering RELP3_main.py for {product} {generation} {failure_mode} with path: {downloaded_folder}")
    
    try:
        # Construct command line arguments
        # We pass arguments as key=value pairs or positional arguments depending on how RELP3_main.py is updated.
        # Based on the plan, we will update RELP3_main.py to accept arguments.
        # Let's use a standard flag approach: --pic_path, --product, --generation, --failure_mode
        
        cmd = [
            sys.executable, 'RELP3_main.py',
            '--pic_path', downloaded_folder,
            '--product', str(product),
            '--generation', str(generation)
        ]
        
        # Only add --failure_mode if not using auto-detect
        if not auto_detect_fm and failure_mode:
            cmd.extend(['--failure_mode', str(failure_mode)])
        
        # Add downloading_radar argument if provided
        if downloading_radar:
            cmd.extend(['--downloading_radar', str(downloading_radar)])
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Capture stdout to find the result folder
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        result_folder_path = None
        
        # Stream output and look for the marker
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip()) # Print to console so we can see progress
                if "RELP_RESULT_FOLDER:" in line:
                    try:
                        result_folder_path = line.split("RELP_RESULT_FOLDER:")[1].strip()
                        print(f"DEBUG: Captured Result Folder from RELP3_main.py: {result_folder_path}")
                    except Exception as e:
                        print(f"Warning: Failed to parse result folder from line: {line}. Error: {e}")

        rc = process.poll()
        if rc != 0:
            print(f"RELP3_main.py exited with error code {rc}")
            # We might still want to return the result folder if it was found before crash?
            # But usually crash means incomplete results.
        else:
            print("RELP3_main.py execution complete.")
            
        return result_folder_path
            
    except subprocess.CalledProcessError as e:
        print(f"Error running RELP3_main.py: {e}")
        return None
    except Exception as e:
        print(f"Error triggering analysis: {e}")
        return None

def upload_results(result_folder, upload_items, uploading_radar_id, failure_mode=None, mapping_key=None):
    """
    Zips specified items from the result folder and uploads to the specified Radar.
    """
    if not upload_items or pd.isna(upload_items) or not uploading_radar_id or pd.isna(uploading_radar_id):
        print("Skipping upload: Upload Items or Uploading Radar not specified.")
        return

    try:
        uploading_radar_id = int(uploading_radar_id)
        items_to_zip = [item.strip() for item in str(upload_items).split(',')]
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate zip filename based on failure mode and mapping key if provided
        if failure_mode and mapping_key:
            # Format: Minor_Wear_Grade B_timestamp.zip
            zip_filename = f"{failure_mode.replace(' ', '_')}_{mapping_key.replace(' ', '_')}_{timestamp_str}.zip"
        elif failure_mode:
            # Format: Minor_Wear_timestamp.zip
            zip_filename = f"{failure_mode.replace(' ', '_')}_{timestamp_str}.zip"
        else:
            # Default format
            zip_filename = f"Analysis_Result_{timestamp_str}.zip"
        
        zip_filepath = os.path.join(result_folder, zip_filename)
        
        print(f"Preparing to zip items: {items_to_zip} into {zip_filename}")
        
        # Create a temporary directory to organize items for zipping
        temp_zip_dir = os.path.join(result_folder, f"temp_zip_{timestamp_str}")
        os.makedirs(temp_zip_dir, exist_ok=True)
        
        items_found = False
        for item in items_to_zip:
            # Handle file extensions if not provided (e.g. Parametric_Output -> Parametric_Output.xlsx)
            # But user said "Parametric_Output名字的文件，因为实际是Parametric_Output.xlsx"
            # So we should look for exact match or match with extension
            
            source_path = os.path.join(result_folder, item)
            if not os.path.exists(source_path):
                # Try adding .xlsx for Parametric_Output
                if item == "Parametric_Output":
                    source_path = os.path.join(result_folder, item + ".xlsx")
            
            if os.path.exists(source_path):
                destination_path = os.path.join(temp_zip_dir, os.path.basename(source_path))
                if os.path.isdir(source_path):
                    shutil.copytree(source_path, destination_path)
                else:
                    shutil.copy2(source_path, destination_path)
                items_found = True
                print(f"Added {item} to zip package.")
            else:
                print(f"Warning: Item '{item}' not found in {result_folder}")

        if not items_found:
            print("No items found to zip. Aborting upload.")
            shutil.rmtree(temp_zip_dir)
            return

        # Create Zip
        shutil.make_archive(os.path.splitext(zip_filepath)[0], 'zip', temp_zip_dir)
        shutil.rmtree(temp_zip_dir) # Cleanup temp dir
        
        print(f"Zip created: {zip_filepath}")
        
        # Upload to Radar
        print(f"Uploading {zip_filename} to Radar {uploading_radar_id}...")
        
        # Ensure authentication
        check_and_renew_kerberos_ticket()
        
        system_identifier = radarclient.ClientSystemIdentifier('RadarClient', '1.0')
        authentication_strategy = radarclient.AuthenticationStrategyAppleConnect()
        client = radarclient.RadarClient(authentication_strategy, client_system_identifier=system_identifier)
        
        radar = client.radar_for_id(uploading_radar_id)
        
        # Check if file already exists on radar (optional, but good practice)
        # But with timestamp, it should be unique.
        
        # Use new_attachment method as seen in RELP_bumper_1click.py
        attachment = radar.new_attachment(zip_filename)
        # attachment.transfer_delegate = ProgressPrintingEnclosureTransferDelegate() # Optional progress delegate
        attachment.set_upload_file(open(zip_filepath, mode='rb'))
        radar.attachments.add(attachment)
        radar.commit_changes()
            
        print(f"Successfully uploaded {zip_filename} to Radar {uploading_radar_id}")

    except Exception as e:
        print(f"Error during upload: {e}")

def monitor_radar_task(radar_id, frequency, file_formats, storage_path, product, generation, failure_mode, multiple_fm_mapping, upload_items, uploading_radar_id):
    """
    Function to be run in a separate thread for each radar task.
    It runs the download logic and then sleeps for 1 hour.
    """
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Starting monitoring cycle...")
        
        downloaded_folder = radar_download(storage_path, radar_id, frequency, file_formats)
        
        if downloaded_folder:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Download complete. Waiting for analysis lock...")
            with analysis_lock:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Acquired analysis lock. Triggering analysis...")
                
                # Process Multiple FM Mapping if provided
                if multiple_fm_mapping and not pd.isna(multiple_fm_mapping):
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Using Multiple FM Mapping: {multiple_fm_mapping}")
                    
                    # Parse Multiple FM Mapping format: [Grade B:Minor_Wear, Grade B:Moderate_Wear]
                    mapping_str = str(multiple_fm_mapping).strip()
                    if mapping_str.startswith('[') and mapping_str.endswith(']'):
                        mapping_str = mapping_str[1:-1]
                    
                    # Split into individual mappings
                    mappings = [m.strip() for m in mapping_str.split(',') if m.strip()]
                    
                    # Group files by failure mode
                    fm_files_map = {}  # {failure_mode: [list of file paths]}
                    fm_to_key_map = {}
                    
                    # First, collect all files (excluding temp folders and zip files)
                    all_files = []
                    for root, dirs, files in os.walk(downloaded_folder):
                        # Skip temp folders created by previous runs
                        dirs[:] = [d for d in dirs if not d.startswith('temp_')]
                        
                        for file in files:
                            # Skip zip files since we've already unzipped them
                            if file.lower().endswith('.zip'):
                                continue
                            
                            file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(file_path, downloaded_folder)
                            all_files.append((file_path, relative_path))
                    
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Found {len(all_files)} files to process")
                    
                    # Process each file to determine which failure mode it belongs to
                    for file_path, relative_path in all_files:
                        # Convert path to lowercase and remove all non-alphanumeric characters for matching
                        normalized_path = ''.join(c.lower() for c in relative_path if c.isalnum())
                        
                        # Check each mapping
                        matched = False
                        for mapping in mappings:
                            if ':' in mapping:
                                key, fm = mapping.split(':', 1)
                                key = key.strip()
                                fm = fm.strip()
                                
                                # Normalize the key for matching
                                normalized_key = ''.join(c.lower() for c in key if c.isalnum())
                                
                                # Check if the normalized path contains the normalized key
                                if normalized_key in normalized_path:
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] File {relative_path} matched key '{key}', assigning to failure mode: {fm}")
                                    
                                    # Add file to the corresponding failure mode group
                                    if fm not in fm_files_map:
                                        fm_files_map[fm] = []
                                        fm_to_key_map[fm] = key
                                    fm_files_map[fm].append(file_path)
                                    matched = True
                                    break
                        
                        if not matched:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Warning: File {relative_path} did not match any mapping key")
                    
                    # Trigger analysis for each failure mode with its specific files
                    result_folders = {}
                    
                    # Generate a base timestamp for all temp folders
                    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # Print summary of files grouped by failure mode
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Summary of files grouped by failure mode:")
                    for fm, file_list in fm_files_map.items():
                        print(f"  - {fm}: {len(file_list)} files")
                        for fp in file_list:
                            print(f"      * {os.path.relpath(fp, downloaded_folder)}")
                    
                    for idx, (fm, file_list) in enumerate(fm_files_map.items()):
                        if not file_list:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Skipping failure mode {fm}: no matching files")
                            continue
                        
                        # Create a temporary folder for this failure mode with unique index
                        temp_fm_folder = os.path.join(downloaded_folder, f"temp_{fm.replace(' ', '_')}_{base_timestamp}_{idx}")
                        os.makedirs(temp_fm_folder, exist_ok=True)
                        
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Created temp folder: {temp_fm_folder}")
                        
                        # Copy matching files to the temporary folder
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Copying {len(file_list)} files to temporary folder for {fm}...")
                        for file_path in file_list:
                            # Preserve relative directory structure
                            relative_path = os.path.relpath(file_path, downloaded_folder)
                            dest_path = os.path.join(temp_fm_folder, relative_path)
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            shutil.copy2(file_path, dest_path)
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Copied: {relative_path}")
                        
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Triggering analysis for failure mode: {fm} with {len(file_list)} files")
                        # Pass the specific failure mode to RELP3_main.py so it only processes that one
                        # This ensures each folder is only processed for its mapped failure mode
                        result_folder = trigger_analysis(temp_fm_folder, product, generation, fm, radar_id, auto_detect_fm=False)
                        if result_folder:
                            result_folders[fm] = result_folder
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Captured result folder for {fm}: {result_folder}")
                            
                            # Wait a bit to ensure all file operations are complete
                            time.sleep(2)
                            
                            # Clean up temporary folder after successful analysis
                            try:
                                if os.path.exists(temp_fm_folder):
                                    shutil.rmtree(temp_fm_folder)
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Cleaned up temporary folder: {temp_fm_folder}")
                            except Exception as e:
                                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Warning: Failed to clean up temporary folder {temp_fm_folder}: {e}")
                        else:
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Warning: No result folder returned for {fm}")
                            # Don't clean up temp folder if analysis failed, for debugging
                            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Keeping temporary folder for debugging: {temp_fm_folder}")
                else:
                    # Use the original failure mode if Multiple FM Mapping is not provided
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Using original Failure Mode: {failure_mode}")
                    result_folder_from_script = trigger_analysis(downloaded_folder, product, generation, failure_mode, radar_id)
                
                # After analysis, try to find the result folder(s)
                if multiple_fm_mapping and not pd.isna(multiple_fm_mapping):
                    # For Multiple FM Mapping, we need to find results for each failure mode
                    if fm_files_map:
                        for fm in fm_files_map.keys():
                            # Use the captured result folder if available
                            if fm in result_folders:
                                found_result_folder = result_folders[fm]
                                if os.path.exists(found_result_folder):
                                    mapping_key = fm_to_key_map.get(fm, None)
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Analysis complete. Uploading results from captured folder {found_result_folder} for failure mode: {fm}...")
                                    upload_results(found_result_folder, upload_items, uploading_radar_id, fm, mapping_key)
                                else:
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Warning: Captured result folder does not exist: {found_result_folder}")
                            else:
                                # Fallback to searching for the result folder
                                # New folder name format: {Product}_{Generation}_{Failure Mode}_{fm}_Result
                                fm_normalized = str(fm).replace(' ', '_')
                                product_normalized = str(product).replace(' ', '_') if product else ""
                                generation_normalized = str(generation).replace(' ', '_') if generation else ""
                                
                                # Try to find result folder with new naming format
                                possible_result_folders = []
                                
                                # New format with Product and Generation
                                if product_normalized and generation_normalized:
                                    possible_result_folders.extend([
                                        os.path.join(os.getcwd(), f"{product_normalized}_{generation_normalized}_{fm_normalized}_{fm_normalized}_Result"),
                                        os.path.join(os.getcwd(), "Result", str(radar_id), f"{product_normalized}_{generation_normalized}_{fm_normalized}_{fm_normalized}_Result"),
                                        os.path.join(os.getcwd(), "result", str(radar_id), f"{product_normalized}_{generation_normalized}_{fm_normalized}_{fm_normalized}_Result"),
                                        os.path.join(os.getcwd(), "Results", str(radar_id), f"{product_normalized}_{generation_normalized}_{fm_normalized}_{fm_normalized}_Result"),
                                    ])
                                
                                # Legacy format (fallback)
                                possible_result_folders.extend([
                                    os.path.join(os.getcwd(), f"{fm_normalized}_Result"),
                                    os.path.join(os.getcwd(), "Result", str(radar_id), f"{fm_normalized}_Result"),
                                    os.path.join(os.getcwd(), "result", str(radar_id), f"{fm_normalized}_Result"),
                                    os.path.join(os.getcwd(), "Results", str(radar_id), f"{fm_normalized}_Result"),
                                ])
                                
                                # Also try to find any folder ending with _{fm}_Result
                                import glob
                                for base_path in [os.getcwd(), os.path.join(os.getcwd(), "Result", str(radar_id)), os.path.join(os.getcwd(), "result", str(radar_id))]:
                                    if os.path.exists(base_path):
                                        pattern = os.path.join(base_path, f"*_{fm_normalized}_Result")
                                        matching_folders = glob.glob(pattern)
                                        if matching_folders:
                                            possible_result_folders.extend(matching_folders)
                                
                                # Remove duplicates while preserving order
                                seen = set()
                                unique_folders = []
                                for folder_path in possible_result_folders:
                                    if folder_path not in seen and os.path.exists(folder_path):
                                        seen.add(folder_path)
                                        unique_folders.append(folder_path)
                                
                                # Check all possible locations
                                found_result_folder = None
                                for folder_path in unique_folders:
                                    if os.path.exists(folder_path):
                                        found_result_folder = folder_path
                                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Found result folder: {found_result_folder}")
                                        break
                                
                                if found_result_folder:
                                    mapping_key = fm_to_key_map.get(fm, None)
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Analysis complete. Uploading results from {found_result_folder} for failure mode: {fm}...")
                                    upload_results(found_result_folder, upload_items, uploading_radar_id, fm, mapping_key)
                                else:
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Warning: Could not find result folder for failure mode: {fm}")
                                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Tried locations: {', '.join(unique_folders[:5])}...")
                    else:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Warning: No files matched any Multiple FM Mapping patterns. Skipping upload.")
                else:
                    # For original failure mode, use the existing logic
                    fm_normalized = str(failure_mode).replace(' ', '_')
                    product_normalized = str(product).replace(' ', '_') if product else ""
                    generation_normalized = str(generation).replace(' ', '_') if generation else ""
                    
                    # Use the captured path if available
                    if result_folder_from_script and os.path.exists(result_folder_from_script):
                        final_result_folder = result_folder_from_script
                    else:
                        # Try to find result folder with new naming format
                        final_result_folder = None
                        
                        # New format with Product and Generation
                        if product_normalized and generation_normalized:
                            new_format_folder = os.path.join(os.getcwd(), "Result", str(radar_id), f"{product_normalized}_{generation_normalized}_{fm_normalized}_{fm_normalized}_Result")
                            if os.path.exists(new_format_folder):
                                final_result_folder = new_format_folder
                        
                        # Fallback to legacy format
                        if not final_result_folder:
                            result_folder_guess = os.path.join(os.getcwd(), "Result", str(radar_id), f"{fm_normalized}_Result")
                            if os.path.exists(result_folder_guess):
                                final_result_folder = result_folder_guess
                            else:
                                # Try glob pattern
                                import glob
                                pattern = os.path.join(os.getcwd(), "Result", str(radar_id), f"*_{fm_normalized}_Result")
                                matching_folders = glob.glob(pattern)
                                if matching_folders:
                                    final_result_folder = matching_folders[0]
                                else:
                                    final_result_folder = os.path.join(os.getcwd(), f"{fm_normalized}_Result")
                    
                    # If that doesn't exist, maybe we can't upload.
                    if final_result_folder and os.path.exists(final_result_folder):
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Analysis complete. Uploading results from {final_result_folder}...")
                        upload_results(final_result_folder, upload_items, uploading_radar_id)
                    else:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Warning: Could not find result folder {final_result_folder} to upload.")
                
                # Rename downloaded folder to indicate analysis completion
                if os.path.exists(downloaded_folder) and not downloaded_folder.endswith("_analyzed"):
                    # Remove _downloaded suffix if present before appending _analyzed
                    base_folder_name = downloaded_folder
                    if base_folder_name.endswith("_downloaded"):
                        base_folder_name = base_folder_name[:-11] # Remove last 11 chars ("_downloaded")
                        
                    analyzed_folder_name = base_folder_name + "_analyzed"
                    try:
                        os.rename(downloaded_folder, analyzed_folder_name)
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Renamed downloaded folder to: {analyzed_folder_name}")
                    except Exception as e:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Warning: Failed to rename folder to _analyzed: {e}")
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Releasing lock.")
        
        # Sleep for 1 hour (3600 seconds) regardless of the frequency parameter
        # The frequency parameter is used to filter files (files uploaded within the last X hours)
        sleep_seconds = 3600
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Radar {radar_id}] Cycle complete. Sleeping for {sleep_seconds} seconds...")
        time.sleep(sleep_seconds)

def main():
    # 激活防休眠锁屏
    prevent_sleep()

    config_file = 'RELP_Configuration.xlsx'
    sheet_name = 'Auto Run'
    
    print(f"Reading configuration from {config_file}, sheet '{sheet_name}'...")
    
    try:
        df = pd.read_excel(config_file, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return

    threads = []

    # Iterate through each row in the configuration and start a thread
    for index, row in df.iterrows():
        try:
            radar_id = row['Downloading_Radar']
            if pd.isna(radar_id):
                continue
            radar_id = int(radar_id)
            
            frequency = row['Frequency(hr)']
            if pd.isna(frequency):
                frequency = 24 # Default to 24 hours if not specified
            
            file_format_str = row['File_format']
            if pd.isna(file_format_str):
                file_formats = []
            else:
                # Split by comma and strip whitespace
                file_formats = [fmt.strip() for fmt in str(file_format_str).split(',')]
            
            base_storage_path = row['File_Storage_Path']
            if pd.isna(base_storage_path):
                base_storage_path = 'download' # Default path
            
            # Construct storage path: base_path/radar_id
            storage_path = os.path.join(str(base_storage_path), str(radar_id))
            
            product = row.get('Product')
            generation = row.get('Generation')
            failure_mode = row.get('Failure Mode')
            
            # New columns
            upload_items = row.get('Upload Items')
            uploading_radar_id = row.get('Uploading_Radar')
            multiple_fm_mapping = row.get('Multiple FM Mapping')

            print(f"Starting thread for Radar {radar_id} (Frequency: {frequency}hr)")
            
            # Create and start a thread for this radar task
            t = threading.Thread(target=monitor_radar_task, args=(radar_id, frequency, file_formats, storage_path, product, generation, failure_mode, multiple_fm_mapping, upload_items, uploading_radar_id))
            t.daemon = True # Allow main program to exit even if threads are running (though we keep main alive below)
            t.start()
            threads.append(t)
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    print(f"Started {len(threads)} monitoring threads.")

    # Keep the main thread alive to allow daemon threads to run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all monitors...")
        sys.exit(0)

if __name__ == "__main__":
    main()

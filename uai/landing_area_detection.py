import numpy as np
import cv2
import sys
import os
np.set_printoptions(threshold=sys.maxsize,linewidth=sys.maxsize)

# === En-boy oranı kontrol fonksiyonu ===
def is_landing_area_visible(box, aspect_ratio_min=0.7, aspect_ratio_max=1.3):
    print("checking for the landing area")
    x1, x2, y1, y2 = box
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if height == 0 or width == 0:
        return False
    ratio = width / height
    print(ratio)
    return aspect_ratio_min <= ratio <= aspect_ratio_max


def check_coordinate_intersection(etiketler, image_width=1920, image_height=1080):
    """Check if UAP/UAI coordinates intersect with other objects
    
    Returns:
        tuple: (has_intersection, is_uap, coordinates, landing_suitable)
        - has_intersection: True if UAP/UAI intersects with other objects
        - is_uap: True if UAP (class_id=1), False if UAI (class_id=0)
        - coordinates: [x1, x2, y1, y2] if intersection found, None otherwise
        - landing_suitable: True if landing area is suitable (no intersections)
    """
    class_box = {}
    flag = 0
    
    # First find UAP (class_id=1) and UAI (class_id=0) coordinates
    for key, value in etiketler.items():
        class_id = int(value[0])
        if class_id in [0, 1]:  # UAP or UAI
            left, top, right, bottom = value[1], value[2], value[3], value[4]
            x1 = int(left * image_width - (right * image_width) / 2)
            x2 = int(left * image_width + (right * image_width) / 2)
            y1 = int(top * image_height - (bottom * image_height) / 2)
            y2 = int(top * image_height + (bottom * image_height) / 2)
            
            if class_id == 1:  # UAP
                class_box['uap'] = [x1, x2, y1, y2]
                flag = 1
            elif class_id == 0:  # UAI
                class_box['uai'] = [x1, x2, y1, y2]
                flag = 1
    
    if flag == 0:
        return False, None, None, False
    
    # Check if UAP/UAI intersects with other objects (class_id not 0 or 1)
    for key, value in etiketler.items():
        class_id = int(value[0])
        if class_id in [0, 1]:  # Skip UAP and UAI
            continue
            
        left, top, right, bottom = value[1], value[2], value[3], value[4]
        x1 = int(left * image_width - (right * image_width) / 2)
        x2 = int(left * image_width + (right * image_width) / 2)
        y1 = int(top * image_height - (bottom * image_height) / 2)
        y2 = int(top * image_height + (bottom * image_height) / 2)
        
        # Check intersection with UAP
        if 'uap' in class_box:
            uap_coords = class_box['uap']
            if ((x1 >= uap_coords[0] and x2 <= uap_coords[1]) and 
                (y1 >= uap_coords[2] and y2 <= uap_coords[3])) or (
                ((uap_coords[0] <= x1 <= uap_coords[1]) or 
                 (uap_coords[0] <= x2 <= uap_coords[1])) and 
                ((uap_coords[2] <= y1 <= uap_coords[3]) or 
                 (uap_coords[2] <= y2 <= uap_coords[3]))):
                return True, True, uap_coords, False  # UAP with intersection - landing not suitable
        
        # Check intersection with UAI
        if 'uai' in class_box:
            uai_coords = class_box['uai']
            if ((x1 >= uai_coords[0] and x2 <= uai_coords[1]) and 
                (y1 >= uai_coords[2] and y2 <= uai_coords[3])) or (
                ((uai_coords[0] <= x1 <= uai_coords[1]) or 
                 (uai_coords[0] <= x2 <= uai_coords[1])) and 
                ((uai_coords[2] <= y1 <= uai_coords[3]) or 
                 (uai_coords[2] <= y2 <= uai_coords[3]))):
                return True, False, uai_coords, False  # UAI with intersection - landing not suitable
    
    # If we get here, we found UAP/UAI but no intersections - landing is suitable
    if 'uap' in class_box:
        return False, True, class_box['uap'], True  # UAP without intersection
    if 'uai' in class_box:
        return False, False, class_box['uai'], True  # UAI without intersection
    
    return False, None, None, False

def get_coordinates(txt_path, image_width=1920, image_height=1080):
    """Get coordinates from model detection or determine if image processing is needed
    
    Returns:
        tuple: (use_model, is_uap, coordinates, landing_suitable)
        - use_model: True if model detection should be used, False for image processing
        - is_uap: True if UAP (class_id=1), False if UAI (class_id=0)
        - coordinates: [x1, x2, y1, y2] if model detection, None if image processing
        - landing_suitable: True if landing area is suitable (no intersections)
    """
    try:
        with open(txt_path, "r") as file:
            content = file.readlines()
            if not content:
                raise ValueError(f"Empty coordinate file: {txt_path}")
            
            # Read all lines and store in dictionary
            etiketler = {}
            for i, line in enumerate(content):
                bilgi = line.strip().split()
                if len(bilgi) >= 5:
                    etiketler[i] = [float(x) for x in bilgi]
            
            # Check for intersections
            has_intersection, is_uap, coords, landing_suitable = check_coordinate_intersection(etiketler, image_width, image_height)
            
            if coords is None:
                raise ValueError(f"No UAP (class_id=1) or UAI (class_id=0) coordinates found in {txt_path}")
            
            # If there's an intersection, use model detection
            # If no intersection, use image processing (return None for coordinates)
            return has_intersection, is_uap, coords if has_intersection else None, landing_suitable
            
    except Exception as e:
        raise ValueError(f"Error reading coordinates from {txt_path}: {str(e)}")

def process_image(img_path, coord_txt_path, uap_threshold=127):
    """Process image using either model detection or image processing
    
    Args:
        img_path: Path to the image file
        coord_txt_path: Path to the coordinate text file
        uap_threshold: Threshold value for UAP images (default: 127)
    """
    # Read and validate image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Get coordinates and determine processing method
    use_model, is_uap, coords, landing_suitable = get_coordinates(coord_txt_path, width, height)
    
    if use_model:
        # Use model detection coordinates
        x_min, x_max, y_min, y_max = coords
        print(f"Using model detection for {'UAP' if is_uap else 'UAI'} (intersection found)")
        print(f"Landing area is {'NOT ' if not landing_suitable else ''}suitable")
    else:
        # Use image processing coordinates
        if not is_uap:  # UAI
            x_min, x_max, y_min, y_max = 744, 896, 470, 620  # UAI default
            print("Using image processing for UAI (no intersection)")
        else:  # UAP
            x_min, x_max, y_min, y_max = 775, 925, 500, 644  # UAP default
            print("Using image processing for UAP (no intersection)")
        print("Landing area is suitable (no objects detected in area)")
    
    if not is_uap:  # UAI
        # UAI processing - use red channel
        b, g, r = cv2.split(img)
        processed_img = r
        # UAI thresholding (fixed at 140)
        kordinata_gore = processed_img[y_min:y_max, x_min:x_max]
        kordinata_gore[kordinata_gore < 140] = 0
        kordinata_gore[kordinata_gore >= 140] = 255
    else:  # UAP
        # UAP processing - use grayscale
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # UAP thresholding (adjustable)
        kordinata_gore = processed_img[y_min:y_max, x_min:x_max]
        _, kordinata_gore = cv2.threshold(kordinata_gore, uap_threshold, 255, cv2.THRESH_BINARY)
    
    return kordinata_gore, (x_min, x_max, y_min, y_max), is_uap, use_model, landing_suitable

def analyze_image(blackandwhite):
    print("🖨️ Full binary image matrix (white=█ black= ):")

    for row in blackandwhite:
        line = ""
        for pixel in row:
            if pixel == 255:
                line += "█"  # White pixel block
            else:
                line += " "  # Black pixel space
        print(line)

    # Step 1: Detect top rows per column
    top_rows_per_column = []
    for col in range(blackandwhite.shape[1]):
        found = None
        for row in range(blackandwhite.shape[0]):
            count = 0
            if blackandwhite[row, col] == 255:
                for r in range(row, min(row + 5, blackandwhite.shape[0])):
                    if blackandwhite[r, col] == 255:
                        count += 1
                if count > 4:
                    found = row
                    break
        top_rows_per_column.append(found)

    # Step 2: Detect bottom rows per column
    down_rows_per_column = []
    for col in range(blackandwhite.shape[1]):
        found = None
        for row in reversed(range(blackandwhite.shape[0])):
            count = 0
            if blackandwhite[row, col] == 255:
                for r in range(max(0, row - 4), row + 1):
                    if blackandwhite[r, col] == 255:
                        count += 1
                if count > 4:
                    found = row
                    break
        down_rows_per_column.append(found)

    def correct_top_curve(arr):
        corrected = arr[:]
        valid = [i for i, v in enumerate(arr) if v is not None]
        if not valid:
            return corrected
        peak_idx = min(valid, key=lambda i: arr[i])

        # 2) Dibine kadar "azalma" kontrolü
        for i in range(1, peak_idx + 1):
            if corrected[i] is None or corrected[i - 1] is None:
                continue
            if corrected[i] > corrected[i - 1]:
                corrected[i] = corrected[i - 1]

        # 3) Dibinden sonra "artış" kontrolü
        for i in range(peak_idx + 1, len(arr)):
            if corrected[i] is None or corrected[i - 1] is None:
                continue
            if corrected[i] < corrected[i - 1]:
                corrected[i] = corrected[i - 1]

        return corrected

    def correct_down_curve(arr):
        corrected = arr[:]
        valid = [i for i, v in enumerate(arr) if v is not None]
        if not valid:
            return corrected
        peak_idx = max(valid, key=lambda i: arr[i])

        # 1) Tepeye kadar "artış" kontrolü
        for i in range(1, peak_idx + 1):
            if corrected[i] is None or corrected[i - 1] is None:
                continue
            if corrected[i] < corrected[i - 1]:
                corrected[i] = corrected[i - 1]

        # 2) Tepeden sonra "azalma" kontrolü
        for i in range(peak_idx + 1, len(arr)):
            if corrected[i] is None or corrected[i - 1] is None:
                continue
            if corrected[i] > corrected[i - 1]:
                corrected[i] = corrected[i - 1]

        return corrected

    # Kullanımı:
    smoothed_top_rows = correct_top_curve(top_rows_per_column)
    smoothed_down_rows = correct_down_curve(down_rows_per_column)

    # Step 5: Build final lists with column info
    top_index_list = []
    for col, row in enumerate(smoothed_top_rows):
        if row is not None:
            top_index_list.append([row, col])
        else:
            top_index_list.append(None)

    down_index_list = []
    for col, row in enumerate(smoothed_down_rows):
        if row is not None:
            down_index_list.append([row, col])
        else:
            down_index_list.append(None)

    # Step 6: Print results
    print("📌 Top and Bottom Pixel Indexes (row, col):")
    for top, bottom in zip(top_index_list, down_index_list):
        if None in [top, bottom]:
            continue
        print(f"Top: {top}  ⬇️  Bottom: {bottom}")

    total_white = total_black = total_pixels = 0
    for top, down in zip(top_index_list, down_index_list):
        if top is None or down is None:
            continue
        start_row = min(top[0], down[0])
        end_row = max(top[0], down[0]) + 1
        col = top[1]
        for value in blackandwhite[start_row:end_row, col]:
            if value == 255:
                total_white += 1
            elif value == 0:
                total_black += 1
            total_pixels += 1

    print(f"Total white pixels between top and bottom: {total_white}")
    print(f"Total black pixels between top and bottom: {total_black}")
    print(f"Total pixels inspected: {total_pixels}")

    return {
        "total_pixels": total_pixels,
        "total_white": total_white,
        "total_black": total_black,
        "black_ratio": total_black/total_pixels if total_pixels > 0 else 0
    }

def process_single_image(image_path, coord_path, uap_threshold=127):
    """Process a single image with its coordinates
    
    Args:
        image_path: Path to the image file
        coord_path: Path to the coordinate file
        uap_threshold: Threshold value for UAP images (default: 127)
    """
    # Read and validate image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Read coordinate file to process all UAP and UAI detections
    try:
        with open(coord_path, "r") as file:
            content = file.readlines()
            if not content:
                raise ValueError(f"Empty coordinate file: {coord_path}")
            
            # Read all coordinates for intersection check
            etiketler = {}
            for i, line in enumerate(content):
                bilgi = line.strip().split()
                if len(bilgi) >= 5:
                    class_id = int(bilgi[0])
                    if class_id in [0, 1]:  # Only process UAP (1) and UAI (0)
                        etiketler[i] = [float(x) for x in bilgi]
            
            if not etiketler:
                raise ValueError(f"No UAP (class_id=1) or UAI (class_id=0) coordinates found in {coord_path}")

            # Process each UAP and UAI detection
            results = []
            for key, value in etiketler.items():
                class_id = int(value[0])
                is_uap = (class_id == 1)
                
                # Get coordinates for this detection
                left, top, right, bottom = value[1], value[2], value[3], value[4]
                x_min = int(left * width - (right * width) / 2)
                x_max = int(left * width + (right * width) / 2)
                y_min = int(top * height - (bottom * height) / 2)
                y_max = int(top * height + (bottom * height) / 2)
                coords = [x_min, x_max, y_min, y_max]

                print(f"\nProcessing {'UAP' if is_uap else 'UAI'} detection:")
                print("checking for landing area")
                
                # === Aspect ratio check ===
                if not is_landing_area_visible(coords):
                    print(f"\n{'UAP' if is_uap else 'UAI'} Results:")
                    print(f"Image: {image_path}")
                    print(f"Coordinates: {coord_path}")
                    print("⛔ Landing area aspect ratio is not suitable — analysis aborted.")
                    print("⚠️  Landing suitability: NOT AVAILABLE due to invalid region shape.")
                    continue

                # Check for intersections with other objects
                has_intersection = False
                for other_key, other_value in etiketler.items():
                    if other_key == key:  # Skip self
                        continue
                    other_class_id = int(other_value[0])
                    if other_class_id not in [0, 1]:  # Check intersection with non-UAP/UAI objects
                        other_left, other_top, other_right, other_bottom = other_value[1:5]
                        other_x_min = int(other_left * width - (other_right * width) / 2)
                        other_x_max = int(other_left * width + (other_right * width) / 2)
                        other_y_min = int(other_top * height - (other_bottom * height) / 2)
                        other_y_max = int(other_top * height + (other_bottom * height) / 2)
                        
                        # Check intersection
                        if ((other_x_min >= x_min and other_x_max <= x_max) and 
                            (other_y_min >= y_min and other_y_max <= y_max)) or (
                            ((x_min <= other_x_min <= x_max) or 
                             (x_min <= other_x_max <= x_max)) and 
                            ((y_min <= other_y_min <= y_max) or 
                             (y_min <= other_y_max <= y_max))):
                            has_intersection = True
                            break

                if has_intersection:
                    print(f"\n{'UAP' if is_uap else 'UAI'} Results (Model Detection):")
                    print(f"Image: {image_path}")
                    print(f"Coordinates: {coord_path}")
                    print(f"Landing area is NOT suitable (objects detected in area)")
                    continue

                # Process the image based on UAP/UAI
                if is_uap:
                    # UAP processing - use grayscale
                    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    roi_based = processed_img[y_min:y_max, x_min:x_max]
                    _, roi_based = cv2.threshold(roi_based, uap_threshold, 255, cv2.THRESH_BINARY)
                else:
                    # UAI processing - use red channel
                    b, g, r = cv2.split(img)
                    processed_img = r
                    roi_based = processed_img[y_min:y_max, x_min:x_max]
                    roi_based[roi_based < 140] = 0
                    roi_based[roi_based >= 140] = 255
                
                # Analyze the image
                detection_results = analyze_image(roi_based)
                
                # Determine landing suitability based on total black pixels
                landing_suitable = detection_results['total_black'] <= 100
                
                # Print results
                print(f"\n{'UAP' if is_uap else 'UAI'} Results (Image Processing):")
                print(f"Image: {image_path}")
                print(f"Coordinates: {coord_path}")
                print(f"Coordinates used: ({x_min}, {x_max}, {y_min}, {y_max})")
                print(f"Total pixels: {detection_results['total_pixels']}")
                print(f"Total white pixels: {detection_results['total_white']}")
                print(f"Total black pixels: {detection_results['total_black']}")
                print(f"Black pixel ratio: {detection_results['black_ratio']:.2%}")
                print(f"Landing area is {'NOT ' if not landing_suitable else ''}suitable (black pixels {'>' if not landing_suitable else '<='} 100)")
                
                # Save processed image
                output_dir = "processed_results"
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_name = os.path.join(output_dir, f"{base_name}_{'uap' if is_uap else 'uai'}_processed.jpg")
                cv2.imwrite(output_name, roi_based)
                print(f"\nProcessed image saved as: {output_name}")
                
                results.append({
                    'type': 'UAP' if is_uap else 'UAI',
                    'coordinates': coords,
                    'results': detection_results,
                    'landing_suitable': landing_suitable
                })
            
            return results
            
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

# Main processing
try:
    # Get image and coordinate paths from user input
    image_path = input("Enter image path: ").strip()
    coord_path = input("Enter coordinate file path: ").strip()
    
    # Verify files exist
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(coord_path):
        raise FileNotFoundError(f"Coordinate file not found: {coord_path}")
    
    # Process the image
    results = process_single_image(image_path, coord_path)
    
    if not results:
        print("\nNo valid UAP or UAI detections were processed.")
    
except FileNotFoundError as e:
    print(f"File Error: {str(e)}")
    sys.exit(1)
except ValueError as e:
    print(f"Error: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    sys.exit(1)

# Optional: Display image

# window_name = "Processed Image"
# cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.imshow(window_name, processed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


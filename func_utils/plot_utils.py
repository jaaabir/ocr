import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import cv2
import numpy as np

def show_image(image, figsize = (12,8)):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')

def render_text_positions(data_dict, img_size, label_key = 'text'):
    """
    Create a black image and render text at specified positions.

    Args:
        data_dict (dict): {'objects': [{'name': 'table', 'bbox': [37.3586, 37.0904, 618.2697, 102.8725]} -> [xmin, ymin, xmax, ymax]
        img_size (tuple): (height, width) of the output image

    Returns:
        np.ndarray: image with rendered text
    """
    # Create a black image
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # white text
    thickness = 1

    # Process each chunk
    for chunk in data_dict.get('objects', []):
        pos = chunk.get('bbox')
        text = chunk.get(label_key, '')
        if not pos or len(pos) != 4:
            continue  # skip invalid positions

        x1, y1, x2, y2 = pos
        x = int(x1)
        y = int((y1 + y2) // 2)

        # Ensure x and y are within image bounds
        if 0 <= x < img_size[1] and 0 <= y < img_size[0] and text != '':
            cv2.putText(img, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)

    return img

def render_bbox(data_dict, image, root_keyname = None, label_keyname = None):
    """
    Create a black image and render text at specified positions.

    Args:
        data_dict (dict): {'objects': [{'name': 'table', 'bbox': [37.3586, 37.0904, 618.2697, 102.8725]} -> [xmin, ymin, xmax, ymax]
        img_size (tuple): (height, width) of the output image

    Returns:
        np.ndarray: image with rendered text
    """
    img = image.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
    # Font settings
    if label_keyname:
        objects = (data_dict[root_keyname] if root_keyname else data_dict)
        elems = [d[label_keyname] for d in objects]
        colors = get_colors_mapper(elems)
    else:
        objects = data_dict
    thickness = 1

    for obj in objects:
        x1,y1,x2,y2 = list(map(int, obj['bbox'])) if label_keyname else list(map(int, obj))
        if label_keyname:
            cv2.rectangle(img, (x1,y1), (x2,y2), colors[obj[label_keyname]], thickness=thickness)
        else:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0,0), thickness=thickness)

    return img

def get_colors_mapper(elems, cmap='tab10'):
    colors = cm.get_cmap(cmap)
    elems= np.unique(elems)
    hmap = {elems[i]: colors(i) for i in range(len(elems))}
    return hmap

def plot_image_vs_render(indexes, tdataset):
    """
    Plots original grayscale images and rendered text overlays side by side.

    Args:
        indexes (list of int): List of indices to visualize
        tdataset (Dataset): PyTorch dataset with 'image' and 'chunks'
    """
    n = len(indexes)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 4 * n))

    # Handle the case when n == 1
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indexes):
        sample = tdataset[idx]
        img_tensor = sample['image']  
        objects = sample['words']    

        # Convert grayscale image tensor to numpy (C=1, H, W) -> (H, W)
        img_np = img_tensor.numpy()

        # Get image shape for render_text_positions
        img_h, img_w = img_np.shape
        rendered = render_text_positions({'objects': objects}, (img_h, img_w))

        # Plot original grayscale image
        ax1 = axes[row, 0]
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title(f"Original Grayscale (idx={idx})")
        ax1.axis('off')

        # Plot rendered RGB image
        ax2 = axes[row, 1]
        ax2.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        ax2.set_title("Rendered Text Positions")
        ax2.axis('off')

    plt.tight_layout()
    plt.show()

def plot_image_and_bbox(indexes, tdataset, bboxes_keyname = None):
    """
    Plots original grayscale images and rendered text overlays side by side.

    Args:
        indexes (list of int): List of indices to visualize
        tdataset (Dataset): PyTorch dataset with 'image' and 'chunks'
    """
    n = len(indexes)
    fig, axes = plt.subplots(nrows=n, ncols=2 if bboxes_keyname else 3, figsize=(15, 4 * n)) 

    # Handle the case when n == 1
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indexes):
        sample = tdataset[idx]
        img_tensor = sample['image']  
        if bboxes_keyname is None:
            words = sample['words']    
            struct = sample['struct']    

        # Convert grayscale image tensor to numpy (C=1, H, W) -> (H, W)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.permute(1,2,0)
        img_np = img_tensor.numpy()

        if len(img_np.shape) == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Get image shape for render_text_positions
        img_h, img_w = img_np.shape
        if bboxes_keyname:
            layout_img = None
            words_img = None
            all_bbox_img = render_bbox(tdataset[idx][bboxes_keyname], img_np) 
        else:
            layout_img = render_bbox(struct, img_np, 'objects')
            words_img = render_bbox(words, img_np, label_keyname='block_num')
            all_bbox_img = None

        # Plot original grayscale image
        ax1 = axes[row, 0]
        ax1.imshow(img_np, cmap='gray')
        ax1.set_title(f"(idx={idx}) | {(img_h, img_w)}")
        ax1.axis('off')

        if bboxes_keyname:
            ax2 = axes[row, 1]
            ax2.imshow(all_bbox_img)
            ax2.set_title("Document Layout Bbox")
            ax2.legend()
            ax2.axis('off')
        else:
            # Plot rendered bbox RGB image
            ax2 = axes[row, 1]
            ax2.imshow(layout_img)
            ax2.set_title("Document Layout Bbox")
            ax2.legend()
            ax2.axis('off')

            # Plot rendered bbox RGB image
            ax2 = axes[row, 2]
            ax2.imshow(words_img)
            ax2.set_title("Document Layout Bbox [text]")
            ax2.axis('off')

    plt.tight_layout()
    plt.show()
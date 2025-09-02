import cv2
import numpy as np

import torch
from torchvision import transforms

from .bisenet import BiSeNet


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='parsing_map_on_im2.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def get_face_mask(face_parser, images, batch_size=128):
    # images: Bx3xHxW
    kernel = np.ones((13, 13), np.float32) 
    face_masks = []
    for i in range(0, images.shape[0], batch_size):
        images_batch = images[i:i+batch_size]
        with torch.no_grad():
            out = face_parser(images_batch)[0]
            parsing = out.cpu().numpy().argmax(1)
            masks = np.zeros_like(parsing, np.float32)
            for idx in range(1, 14):
                masks[parsing == idx] = 1
            
            for mask in masks:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                mask = cv2.dilate(mask, kernel, iterations=3)
                face_masks.append(mask)

    return face_masks


def build_face_parser(weight_path, resnet_weight_path, n_classes=19, device_id=0):
    model_state_dict = torch.load(weight_path, weights_only=False)
    bisenet = BiSeNet(n_classes, resnet_weight_path=resnet_weight_path)
    # load model
    #bisenet.load_state_dict(model_state_dict, strict=True)
    bisenet_state_dict = bisenet.state_dict()
    for k, v in model_state_dict.items():
        if 'fc' in k: continue
        bisenet_state_dict.update({k: v})
    bisenet.load_state_dict(bisenet_state_dict)
    bisenet.to(f"cuda:{device_id}")

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return bisenet.eval(), to_tensor




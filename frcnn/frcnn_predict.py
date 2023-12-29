# from mmdet.apis import init_detector, inference_detector

# # config_file = 'faster-rcnn_r101_fpn_1x_coco.py'
# # checkpoint_file = 'faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
# # model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
# # res = inference_detector(model, 'dog.jpeg')
from mmdetection.mmdet.apis import DetInferencer
def frcnnpredict(img,weight):
    # Initialize the DetInferencer
    inferencer = DetInferencer(model= r'frcnn\model\faster-rcnn_r101_fpn_1x_coco.py', weights=weight)
    res =inferencer(img, out_dir='outputs/', no_save_pred=False)
    return res['visualization']
# print(frcnnpredict('Japan_000082.jpg',r'frcnn\checkpoint\best_coco_bbox_mAP_epoch_20.pth'))
# Initialize the DetInferencer
# inferencer = DetInferencer(model= r'frcnn\model\faster-rcnn_r101_fpn_1x_coco.py', weights=r'frcnn\checkpoint\best_coco_bbox_mAP_epoch_20.pth')
# res =inferencer('Japan_000082.jpg',out_dir='outputs/', no_save_pred=False)
# print(res)
# breakpoint()
from .deep_sort import DeepSort
from utils.logger import log


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(model, cfg, use_cuda):
    log(f'Configuration {cfg}')
    return DeepSort(
        model,
        input_size_h=cfg["INPUT_SIZE_H"],
        input_size_w=cfg["INPUT_SIZE_W"],
        max_dist=cfg["MAX_DIST"],
        min_confidence=cfg["MIN_CONFIDENCE"], 
        nms_max_overlap=cfg["NMS_MAX_OVERLAP"], 
        max_iou_distance=cfg["MAX_IOU_DISTANCE"], 
        max_age=cfg["MAX_AGE"], 
        n_init=cfg["N_INIT"], 
        nn_budget=cfg["NN_BUDGET"], 
        use_cuda=use_cuda)


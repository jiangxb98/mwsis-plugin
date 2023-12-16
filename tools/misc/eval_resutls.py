import torch
import json
from pycocotools import mask
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8');
        return json.JSONEncoder.default(self, obj)

class COCOeval_(COCOeval):
    def summarize(self, catId=None):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.4f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if isinstance(catId, int):
                    s = s[:, :, catId, aind, mind]
                elif isinstance(catId, list):
                    s = s[:, :, int(catId[0]):int(catId[1])+1, aind, mind]
                else:
                    s = s[:, :, :, aind, mind]
                # s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                # s = s[:,:,aind,mind]
                if isinstance(catId, int):
                    s = s[:,catId, aind, mind]
                elif isinstance(catId, list):
                    s = s[:, int(catId[0]):int(catId[1])+1, aind, mind]
                else:
                    s = s[:, :, aind, mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if isinstance(catId, int):
            class_name = ['Car','Ped','Cyc']
            print('Eval {} Results:'.format(class_name[catId]))
        if isinstance(catId, list):
            print('Eval MAP Results:')            
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()
        return self.stats


if __name__ == '__main__':
    # results_mask_3d_path = '/home/jiangguangfeng/桌面/codebase/mask3d_preds.json'
    results_mask_3d_path = '/root/3D/work_dirs/dataset_infos/mask3d_preds.json'
    validation_path = '/root/3D/work_dirs/dataset_infos/validation_3dmask_infos.json'
    # validation_path = '/home/jiangguangfeng/桌面/codebase/validation_3dmask_infos_front.json'
    annType = ['segm','bbox','keypoints']
    annType = annType[1]  # specify type here
    cocoGt = COCO(validation_path)
    cocoDt = cocoGt.loadRes(results_mask_3d_path)
    cocoEval = COCOeval_(cocoGt, cocoDt,annType)
    # cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    # cocoEval.summarize()
    result2 = cocoEval.summarize(catId=[0,1])
    result1 = cocoEval.summarize(catId=0)
    result2 = cocoEval.summarize(catId=1)